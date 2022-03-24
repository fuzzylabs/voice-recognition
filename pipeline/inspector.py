from zenml.repository import Repository

repo = Repository()
p = repo.get_pipeline(pipeline_name="train_evaluate_and_deploy_pipeline")
runs = p.runs
print(f"Pipeline `train_evaluate_and_deploy_pipeline` has {len(runs)} run(s)")
run = runs[-1]
print(f"The run you just made has {len(run.steps)} steps.")
step = run.get_step('keras_evaluator')
print(
    f"The `keras_evaluator step` returned an accuracy: {step.outputs['accuracy'].read()}"
)