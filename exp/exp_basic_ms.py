import os
import mindspore as ms
import mindspore.context as context

class Exp_Basic:
    def __init__(self, args):
        self.args = args
        self._set_device()
        self.model = self._build_model()

    def _set_device(self):
        if self.args.use_gpu:
            context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU", device_id=self.args.gpu)
            # context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=self.args.gpu)
            # context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend", device_id=self.args.gpu)
            print(f"Using GPU: {self.args.gpu}")
        else:
            context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
            print("Using CPU")

    def _build_model(self):
        raise NotImplementedError

    def _get_data(self, *args, **kwargs):
        pass

    def vali(self, *args, **kwargs):
        pass

    def train(self, *args, **kwargs):
        pass

    def test(self, *args, **kwargs):
        pass
