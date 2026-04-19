class EMA:
    def __init__(self, model, decay=0.999, device=None):
        self.decay = decay
        self.model = model
        if device is None:
            device = next(model.parameters()).device
        self.device = device
        self.shadow = {}
        self.backup = {}

        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                if self.shadow[name].device != param.data.device:
                    self.shadow[name] = self.shadow[name].to(param.data.device)
                new_avg = self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                self.shadow[name] = new_avg.clone()

    def apply_shadow(self):
        """Use EMA weights (for sampling/eval)"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                if self.shadow[name].device != param.data.device:
                    self.shadow[name] = self.shadow[name].to(param.data.device)
                param.data = self.shadow[name]

    def restore(self):
        """Restore original weights after eval"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}