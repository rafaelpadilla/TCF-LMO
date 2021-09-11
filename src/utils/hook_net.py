
class Hook():
    def __init__(self, module, backwards=False):
        self.backwards = backwards
        # It is a backward hook
        if backwards:
            self.hook = module.register_backward_hook(self.hook_fn)
        # It is a forward hook
        else:
            self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, module_input, module_output):
        self.input = module_input
        self.output = module_output
