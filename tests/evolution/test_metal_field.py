"""Driven electrodes: state ownership, thermodynamic work and time integration."""
import unittest
from unittest.mock import patch
import torch
from equicdft import MetalWall, GridCACEModel
from equicdft.functional.base import ReadoutContext
from equicdft.evolution import (RampedField, ExternalField, Dynamics, AdaptiveDynamics,
    EulerIntegrator, SemiImplicitIntegrator, StochasticEulerIntegrator, ConstantMobility, StepControl)
from equicdft.evolution.field import _at_time, _FieldEvaluator, _work, _same_fields
from tests.metal_helpers import liquid_coulomb, metal_sites
from tests.functional.test_metalwall import _field


class MetalModel(torch.nn.Module):
    """Convex local liquid plus actual reciprocal MetalWall, with exact derivatives."""
    def __init__(self, data):
        super().__init__()
        self.grid_spacing = data['grid_spacing']
        self.mean_temperature = self.boltzmann_constant = 1.
        self.wall = MetalWall(liquid_coulomb((1.,-1.),1.7),metal_sites(data),.35).double()

    def context(self, data):
        return ReadoutContext.from_data(data,grid_spacing=self.grid_spacing)

    def forward(self,data,**kwargs):
        rho=data['rho'].requires_grad_(True)
        ctx=self.context(data)
        energy,outputs=self.wall.energy_and_outputs(ctx)
        energy=energy+10*ctx.voxel_volume*rho.square().sum()
        derivative=torch.autograd.grad(energy,rho,create_graph=True)[0]/ctx.voxel_volume
        return dict(beta_F=energy,density_derivative=derivative,**outputs)


class TestMetalField(unittest.TestCase):
    def setUp(self):
        self.dtype=torch.get_default_dtype(); torch.set_default_dtype(torch.float64)
        data=_field(shape=(4,2,2),group_ids=(0,0))
        data.update(metal_group_ids=torch.tensor([0]),metal_total_charge=torch.zeros(1),
                    temperature=torch.tensor(2.),beta=torch.tensor(.5))
        data['grid_positions']=torch.cartesian_prod(torch.arange(4),torch.arange(2),torch.arange(2))
        data['V_ext']=torch.zeros_like(data['rho'])
        self.data=data; self.model=MetalModel(data)
        final=-.2*data['grid_positions'][:,0,None]*data['grid_spacing'][0]*torch.tensor([1.,-1.])
        self.field=RampedField(data['V_ext'],final,.02,
                              initial_metal_field=[0.,0.,0.],final_metal_field=[.2,0.,0.])

    def tearDown(self):
        torch.set_default_dtype(self.dtype)

    def stepper(self, kind=EulerIntegrator, mobility=.001):
        options=dict(reference_density=[.4,.4],reference_curvature=20*torch.eye(2)) if kind is SemiImplicitIntegrator else {}
        if kind is StochasticEulerIntegrator: options['seed']=7
        return kind(self.model,ConstantMobility(mobility),1e-3,**options)

    def test_override_matches_static_constructor_and_reuses_cache(self):
        a=_at_time(self.field,.02,self.data)
        wall=self.model.wall
        actual=wall(self.model.context(a)); cached=wall._cache
        wall.external_field.copy_(a['metal_external_field'])
        static=wall(self.model.context(self.data))
        for key in actual: torch.testing.assert_close(actual[key],static[key])
        wall.external_field.zero_()
        wall(self.model.context(_at_time(self.field,0,self.data)))
        self.assertIs(cached,wall._cache)
        self.assertTrue(torch.equal(wall.external_field,torch.zeros(3)))

    def test_full_model_driven_field_and_evolution(self):
        # Use the public model directly: oracle adapters strip electrode keys.
        data={k:v for k,v in self.data.items() if not k.startswith('metal_')}
        wall=self.model.wall
        model=GridCACEModel(None,None,[wall],grid_spacing=data['grid_spacing'],
                            boltzmann_constant=1.,compute_local_mu=False).double().eval()
        driven=_at_time(self.field,.02,data)
        actual=model(dict(driven,rho=driven['rho'].clone()))
        wall.external_field.copy_(driven['metal_external_field'])
        static=model(dict(data,rho=data['rho'].clone(),V_ext=driven['V_ext']))
        for key in ('beta_F','density_derivative','metal_site_q'):
            torch.testing.assert_close(actual[key],static[key])
        wall.external_field.zero_()
        frozen=dict(data,mask=torch.ones(len(data['rho']),dtype=torch.bool))
        for kind in (EulerIntegrator,SemiImplicitIntegrator):
            options=dict(reference_density=[.4,.4],reference_curvature=20*torch.eye(2)) if kind is SemiImplicitIntegrator else {}
            stepper=kind(model,ConstantMobility(.001),1e-3,**options)
            result=AdaptiveDynamics(stepper,StepControl(1e-8,.005),self.field).run(frozen,.03)
            self.assertTrue(result['metadata']['completed'])
            torch.testing.assert_close(result['beta_A']-result['beta_A'][0],result['beta_work'],atol=1e-12,rtol=1e-12)
        self.assertTrue(torch.equal(wall.external_field,torch.zeros(3)))
        with self.assertRaisesRegex(ValueError,'geometry'):
            model(dict(data,metal_positions=wall.metal_positions))

    def test_work_exact_at_fixed_density_includes_induced_charge(self):
        a=_at_time(self.field,0,self.data); b=_at_time(self.field,.02,self.data)
        integ=self.stepper()
        work=_work(a,b,self.model)
        torch.testing.assert_close(work,integ.beta_A(b)-integ.beta_A(a),atol=1e-12,rtol=1e-12)
        liquid_only=_work(dict(a,metal_external_field=a['metal_external_field']),a,self.model)
        self.assertEqual(float(liquid_only),0.)
        h=1e-6
        middle=_at_time(self.field,.01,self.data)
        plus=dict(middle,metal_external_field=middle['metal_external_field']+torch.tensor([h,0.,0.]))
        minus=dict(middle,metal_external_field=middle['metal_external_field']-torch.tensor([h,0.,0.]))
        fd=(integ.beta_A(plus)-integ.beta_A(minus))/(2*h)
        derivative=self.model.wall.external_field_work(self.model.context(middle),torch.tensor([1.,0.,0.]))
        torch.testing.assert_close(fd,derivative,atol=1e-8,rtol=1e-8)

    def test_frozen_liquid_ramp_work_all_drivers(self):
        frozen=dict(self.data,mask=torch.ones(len(self.data['rho']),dtype=torch.bool))
        for kind in (EulerIntegrator,SemiImplicitIntegrator):
            for adaptive in (False,True):
                integ=self.stepper(kind)
                if adaptive:
                    driver=AdaptiveDynamics(integ,StepControl(1e-8,.005),self.field)
                    result=driver.run(frozen,.03)
                else: result=Dynamics(integ,self.field).run(frozen,30)
                torch.testing.assert_close(result['rho'],result['rho'][0].expand_as(result['rho']),rtol=0,atol=0)
                torch.testing.assert_close(result['beta_A']-result['beta_A'][0],result['beta_work'],atol=1e-12,rtol=1e-12)
                for t,e in zip(result['time'],result['metal_external_field']):
                    torch.testing.assert_close(e,self.field(float(t),frozen)['metal_external_field'])

    def test_restart_and_density_motion(self):
        full=Dynamics(self.stepper(),self.field).run(self.data,30)
        first=Dynamics(self.stepper(),self.field).run(self.data,12)
        state=dict(self.data,rho=first['rho'][-1],metal_external_field=first['metal_external_field'][-1])
        rest=Dynamics(self.stepper(),self.field).run(state,18,start_time=.012)
        torch.testing.assert_close(rest['rho'][-1],full['rho'][-1],atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(rest['beta_work'][-1]+first['beta_work'][-1],full['beta_work'][-1])
        self.assertTrue(torch.all(full['rho'][:,self.data['excluded_mask']]==0))
        torch.testing.assert_close(full['particle_numbers'],full['particle_numbers'][0].expand_as(full['particle_numbers']))
        self.assertTrue(torch.equal(self.model.wall.external_field,torch.zeros(3)))

    def test_electrode_only_switch_and_adaptive_rejection(self):
        class Switch(ExternalField):
            breakpoints=(.003,)
            def __call__(self,t,data):
                return dict(V_ext=data['V_ext'],metal_external_field=data['rho'].new_tensor([.2 if t>=.003 else 0.,0.,0.]))
        result=AdaptiveDynamics(self.stepper(),StepControl(1e-8,.01,error_tolerance=1e-7),Switch()).run(self.data,.02)
        self.assertIn(.003,result['time'].tolist())
        corrected=result['beta_A']-result['beta_work']
        self.assertTrue(torch.all(torch.diff(corrected)<1e-10))
        fine=Dynamics(self.stepper(),Switch()); fine.integrator.dt=1e-5
        reference=fine.run(self.data,2000,save_every=2000)
        torch.testing.assert_close(result['rho'][-1],reference['rho'][-1],atol=1e-6,rtol=1e-5)

    def test_excluded_semiimplicit_motion_and_refinement(self):
        control=StepControl(1e-9,.01,error_tolerance=1e-6)
        result=AdaptiveDynamics(self.stepper(SemiImplicitIntegrator),control,self.field).run(self.data,.03)
        fine=self.stepper(); fine.dt=1e-5
        reference=Dynamics(fine,self.field).run(self.data,3000,save_every=3000)
        torch.testing.assert_close(result['rho'][-1],reference['rho'][-1],atol=3e-7,rtol=2e-6)
        self.assertTrue(torch.all(result['rho'][:,self.data['excluded_mask']]==0))
        torch.testing.assert_close(result['particle_numbers'],result['particle_numbers'][0].expand_as(result['particle_numbers']),atol=1e-12,rtol=0)
        self.assertTrue(torch.all(torch.diff(result['beta_A']-result['beta_work'])<1e-10))

    def test_invalid_fields_and_groups_fail(self):
        with self.assertRaises(ValueError):RampedField(self.data['V_ext'],self.data['V_ext'],1,initial_metal_field=[0,0,0])
        self.model.wall.metal_group_ids=torch.tensor([0,1])
        with self.assertRaisesRegex(ValueError,'exactly one'):
            self.model.wall(self.model.context(_at_time(self.field,.02,self.data)))

    def test_cached_work_matches_midpoint_without_extra_solves(self):
        a=_at_time(self.field,0,self.data)
        b=_at_time(self.field,.02,dict(self.data,rho=self.data['rho']*1.01))
        integ=self.stepper()
        integ.rhs(a); pa=integ.last_metal_dipole
        integ.rhs(b); pb=integ.last_metal_dipole
        expected=_work(a,b,self.model)
        with patch.object(self.model.wall,'energy_and_outputs',side_effect=AssertionError('extra solve')):
            actual=_work(a,b,self.model,start_dipole=pa,end_dipole=pb)
        torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-12)
        self.assertFalse(pa.requires_grad)

    def test_adaptive_cache_matches_fallback_after_rejections(self):
        from equicdft.evolution import adaptive
        control=StepControl(1e-10,.01,error_tolerance=1e-9)
        def run():
            stepper=self.stepper(); stepper.dt=.01
            return AdaptiveDynamics(stepper,control,self.field).run(self.data,.02)
        with patch.object(self.model.wall,'external_field_work',wraps=self.model.wall.external_field_work) as calls:
            cached=run(); cached_calls=calls.call_count
        def fallback(a,b,model,**unused):return _work(a,b,model)
        with patch.object(adaptive,'_work',side_effect=fallback), patch.object(
                self.model.wall,'external_field_work',wraps=self.model.wall.external_field_work) as calls:
            reference=run(); fallback_calls=calls.call_count
        self.assertGreater(cached['metadata']['rejected_accuracy'],0)
        self.assertLess(cached_calls,fallback_calls)
        for key in ('rho','beta_A','beta_work'):
            torch.testing.assert_close(cached[key][-1],reference[key][-1],rtol=1e-10,atol=1e-12)

    def test_charged_group_retains_work_energy_identity(self):
        self.model.wall.metal_total_charge.fill_(.1)
        data=dict(self.data,rho=self.data['rho'].clone())
        free=~data['excluded_mask']
        data['rho'][free,0]-=.1/(free.sum()*data['grid_spacing'].prod())
        a=_at_time(self.field,0,data); b=_at_time(self.field,.02,data)
        integ=self.stepper()
        integ.rhs(a); pa=integ.last_metal_dipole
        integ.rhs(b); pb=integ.last_metal_dipole
        actual=_work(a,b,self.model,start_dipole=pa,end_dipole=pb)
        torch.testing.assert_close(actual,integ.beta_A(b)-integ.beta_A(a),atol=1e-12,rtol=1e-12)

    def test_electric_constructor_matches_manual_three_species_field(self):
        base=torch.ones(2,3); positions=torch.tensor([[1.,2.,3.],[-1.,1.,2.]])
        charges=torch.tensor([2.,0.,-1.]); final=torch.tensor([.1,-.2,.3])
        field=RampedField.from_electric_field(base,positions,charges,2.,final_field=final)
        state={'rho':base}
        for t in (0.,1.,2.,3.):
            out=field(t,state); e=min(t/2,1)*final
            torch.testing.assert_close(out['V_ext'],base-(positions@e)[:,None]*charges)
            torch.testing.assert_close(out['metal_external_field'],e)
        liquid=RampedField.from_electric_field(base,positions,charges,2.,final_field=final,metal=False)
        torch.testing.assert_close(liquid(1.,state),field(1.,state)['V_ext'])
        with self.assertRaisesRegex(ValueError,'grid_center'):
            RampedField.from_electric_field(base,positions[:1],charges,2.,final_field=final)

    def test_missing_keys_callables_and_beta_fail_clearly(self):
        class Intermittent(ExternalField):
            def __call__(self,t,data):
                return dict(V_ext=data['V_ext'],metal_external_field=torch.zeros(3)) if t==0 else data['V_ext']
        field_at=_FieldEvaluator(Intermittent())
        first=field_at(0,self.data)
        with self.assertRaisesRegex(ValueError,'same keys'):field_at(.1,first)
        with self.assertRaisesRegex(ValueError,'both endpoints'):_same_fields(first,self.data)
        end=_at_time(self.field,.02,self.data)
        with self.assertRaisesRegex(ValueError,'torch.nn.Module'):_work(first,end,lambda data: {})
        wrong=dict(first,beta=torch.tensor(1.))
        with self.assertRaisesRegex(ValueError,'fixed, positive beta'):_work(wrong,end,self.model)
        with self.assertRaisesRegex(ValueError,'temperature/k_B'):
            _work(wrong,dict(end,beta=wrong['beta']),self.model)
        with self.assertRaisesRegex(ValueError,'temperature/k_B'):self.stepper().rhs(wrong)

    def test_fixed_cache_matches_midpoint_at_every_saved_frame(self):
        from equicdft.evolution import dynamics
        def fallback(a,b,model,**unused): return _work(a,b,model)
        for kind in (EulerIntegrator, SemiImplicitIntegrator, StochasticEulerIntegrator):
            def run(): return Dynamics(self.stepper(kind),self.field).run(self.data,30,save_every=4)
            with patch.object(self.model.wall,'external_field_work',wraps=self.model.wall.external_field_work) as calls:
                cached=run()
                # A continuous kink may differ by an ulp between its limits;
                # it is not a per-step electrode evaluation.
                self.assertLessEqual(calls.call_count,2)
            with patch.object(dynamics,'_work',side_effect=fallback): reference=run()
            for key in ('rho','beta_A','beta_work'):
                torch.testing.assert_close(cached[key],reference[key],rtol=1e-11,atol=1e-12)
            if 'rng_state' in cached:
                torch.testing.assert_close(cached['rng_state'],reference['rng_state'],rtol=0,atol=0)

    def test_fixed_cache_handles_jump_and_zero_steps(self):
        from equicdft.evolution import dynamics
        class Jump(ExternalField):
            breakpoints=(.0025,)
            def __call__(self,t,data):
                return dict(V_ext=torch.zeros_like(data['rho']),
                            metal_external_field=data['rho'].new_tensor([.2 if t>=.0025 else 0.,0.,0.]))
        def run(n): return Dynamics(self.stepper(),Jump()).run(self.data,n,save_every=4)
        def fallback(a,b,model,**unused): return _work(a,b,model)
        for n in (0,7):
            cached=run(n)
            with patch.object(dynamics,'_work',side_effect=fallback): reference=run(n)
            for key in ('rho','beta_A','beta_work'):
                torch.testing.assert_close(cached[key],reference[key],rtol=1e-11,atol=1e-12)

    def test_fixed_failed_step_settles_pending_work(self):
        from equicdft.evolution import dynamics
        def run():
            stepper=self.stepper()
            step=stepper.step
            count=0
            def failing_step(state):
                nonlocal count
                count+=1
                if count==4: raise ValueError('injected failure before evaluation')
                return step(state)
            with patch.object(stepper,'step',side_effect=failing_step):
                with self.assertRaisesRegex(ValueError,'injected failure') as caught:
                    Dynamics(stepper,self.field).run(self.data,10,save_every=2)
            self.assertEqual(stepper.dt,.001)
            return caught.exception.trajectory
        cached=run()
        def fallback(a,b,model,**unused): return _work(a,b,model)
        with patch.object(dynamics,'_work',side_effect=fallback): reference=run()
        completed=Dynamics(self.stepper(),self.field).run(self.data,3,save_every=2)
        self.assertFalse(cached['metadata']['completed'])
        self.assertEqual(cached['metadata']['steps_completed'],3)
        for key in ('rho','beta_A','beta_work'):
            torch.testing.assert_close(cached[key],reference[key],rtol=1e-11,atol=1e-12)
            torch.testing.assert_close(cached[key],completed[key],rtol=1e-11,atol=1e-12)

    def test_field_key_tracking_stays_out_of_data_and_resets_between_runs(self):
        class Intermittent(ExternalField):
            def __call__(self,t,data):
                return dict(V_ext=data['V_ext'],metal_external_field=torch.zeros(3)) if t==0 else data['V_ext']
        original=self.model.forward
        def checked(state,**kwargs):
            self.assertNotIn('_external_field_keys',state)
            return original(state,**kwargs)
        for adaptive in (False,True):
            driver=(AdaptiveDynamics(self.stepper(),StepControl(1e-8,.005),self.field) if adaptive
                    else Dynamics(self.stepper(),self.field))
            duration=.002 if adaptive else 2
            with patch.object(self.model,'forward',side_effect=checked):
                result=driver.run(self.data,duration)
                driver.field=RampedField(self.data['V_ext'],self.data['V_ext'],.02)
                driver.run(self.data,duration)
                driver.field=Intermittent()
                with self.assertRaisesRegex(ValueError,'same keys') as caught:
                    driver.run(self.data,duration)
            for record in (result,caught.exception.trajectory):
                self.assertNotIn('_external_field_keys',record)
                self.assertNotIn('_external_field_keys',record['metadata'])
        self.assertNotIn('_external_field_keys',self.data)


if __name__=='__main__':unittest.main()
