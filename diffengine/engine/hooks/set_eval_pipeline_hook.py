from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmengine.runner import Runner


@HOOKS.register_module()
class SetEvalPipelineHook(Hook):
    """Set pipeline before val."""

    priority = "NORMAL"

    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        self.kwargs = kwargs

    def before_val(self, runner: Runner, **kwargs) -> None:
        """Before val."""
        runner.model.eval_pipeline_kwargs = self.kwargs

    def before_val_epoch(self, runner: Runner, **kwargs) -> None:
        """Before val epoch."""
        runner.model.set_pipeline()

    def after_val_epoch(self, runner: Runner, **kwargs) -> None:
        """After val epoch."""
        runner.model.del_pipeline()

    def after_val(self, runner: Runner, **kwargs) -> None:
        """After val."""
        del runner.model.eval_pipeline_kwargs

    def before_test(self, runner: Runner, **kwargs) -> None:
        """Before test."""
        runner.model.eval_pipeline_kwargs = self.kwargs

    def before_test_epoch(self, runner: Runner, **kwargs) -> None:
        """Before test epoch."""
        runner.model.set_pipeline()

    def after_test_epoch(self, runner: Runner, **kwargs) -> None:
        """After test epoch."""
        runner.model.del_pipeline()

    def after_test(self, runner: Runner, **kwargs) -> None:
        """After test."""
        del runner.model.eval_pipeline_kwargs
