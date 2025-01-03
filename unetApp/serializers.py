# from rest_framework import serializers
# from .models import Images

# class ImageSerializer(serializers.HyperlinkedModelSerializer):
#     class Meta:
#         model = Images
#         fields = ['unMaskedImage']

from rest_framework import serializers
from .models import Images

class ImageSerializer(serializers.HyperlinkedModelSerializer):
    unMaskedImageUrl = serializers.SerializerMethodField()
    predictedMaskUrl = serializers.SerializerMethodField()

    class Meta:
        model = Images
        fields = ['unMaskedImage', 'predictedMask', 'unMaskedImageUrl', 'predictedMaskUrl']

    def get_unMaskedImageUrl(self, obj):
        return self.context['request'].build_absolute_uri(obj.unMaskedImage.url).replace('example.com', 'localhost')

    # def get_predictedMaskUrl(self, obj):
    #     return self.context['request'].build_absolute_uri(obj.predictedMask.url).replace('example.com', 'localhost')
    
    def get_predictedMaskUrl(self, obj):
        predicted_mask_url = self.context['request'].build_absolute_uri(obj.predictedMask.url).replace('example.com', 'localhost')
        print("Predicted Mask URL:", predicted_mask_url)
        return predicted_mask_url

