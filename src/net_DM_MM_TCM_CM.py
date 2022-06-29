from .abc_pipeline import ABC_Pipeline


class Pipeline(ABC_Pipeline):
    def inference_validation_test(self, tensors):
        # DM
        # batch, 256, 81, 144
        out = self.dissimilarity_module(tensors['feat_ref'], tensors['feat_tar'])
        # batch, 81, 144
        if out.dim() == 2:
            out = out.unsqueeze(0)
        # 81, 144
        # MM (opening)
        if hasattr(self, 'opening'):
            out = self.opening(out)
        # 1, 1, 81, 144
        # MM (closing)
        if hasattr(self, 'closing'):
            out = self.closing(out)
        # TCM
        if hasattr(self, 'temporal_consistency'):
            out = self.temporal_consistency.inference_validation_test(out)
        # 1, 1, 81, 144
        # Count pixels "on" in each image
        out = self.sum_pixels_on(out)
        # 1, 1
        # Now we have the amount of pixels on, we will transform it linearly to have a range between 0 and 1
        out = self.change_scale_for_classification(out)
        # 1, 1
        # CM
        out = self.classification_function(out)
        # 1, 1
        return out

    def forward(self, tensors):
        # DM
        out = self.dissimilarity_module(tensors['feat_ref'], tensors['feat_tar'])
        # batch, 81, 144
        assert out.min().item() >= 0.
        assert out.max().item() <= 1.
        # Train DM
        if tensors['cycle_name'] == 'training DM':
            if len(out.shape) == 2:
                return out.unsqueeze(0).unsqueeze(0)
            else:
                return out.unsqueeze(1)
        # batch, 81, 144
        # MM (opening)
        if hasattr(self, 'opening'):
            out = self.opening(out)
        # batch, 1 81, 144
        # MM (closing)
        if hasattr(self, 'closing'):
            out = self.closing(out)
        # batch, 1, 81, 144
        if tensors['cycle_name'] == 'training MM':
            return out
        # Train TCM
        if tensors['cycle_name'] == 'training TCM':
            self.temporal_consistency.gather_many_train(out)
            return self.temporal_consistency.train()  # dicionário com tensores
        # During training of CM, apply temporal consistency and store frames til we have a full batch
        if tensors['cycle_name'] in ['training CM']:
            # 15, 81, 144
            # TCM
            if hasattr(self, 'temporal_consistency'):
                out = self.temporal_consistency.inference_train(out.squeeze())
                # 81, 144
                # Add temporal consistency result in the list
                self.temporal_consistency.gather_one_by_one_inference(out)
                # Check if the buffer is filled with a full batch
                if self.temporal_consistency.is_buffer_complete():
                    out = self.temporal_consistency.frames_inference.clone().to(self.device)
                # batch, 81, 144
                else:
                    return 'buffer not full yet'
        # Count pixels "on" in each image
        out = self.sum_pixels_on(out)
        # batch, 1
        # Now we have the amount of pixels on, we will transform it linearly to have a range between 0 and 1
        out = self.change_scale_for_classification(out)
        # batch, 1
        # Apply a threshold to have the final classification
        out = self.classification_function(out)
        # batch, 1
        for i, o in enumerate(out):
            if o.item() < 0:
                out[i] += (-1. * o)
                print('negativo')
            if o.item() > 1:
                out[i] += (1. - o)
                print('maior que 1')
        for i in out:
            assert i.item() >= 0 and i.item() <= 1
        return out
