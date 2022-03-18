from zenml.repository import Repository

repo = Repository()
p = repo.get_pipeline(pipeline_name="train_and_evaluate_pipeline")
runs = p.runs
print(f"Pipeline `train_and_evaluate_pipeline` has {len(runs)} run(s)")
run = [run for run in runs if run.name == "train_and_evaluate_pipeline-16_Mar_22-16_34_20_069739"][0]
print(f"The run you just made has {len(run.steps)} steps.")
step = run.get_step('lstm_trainer')
print(
    f"The `lstm_trainer step` had a config: {step}"
)