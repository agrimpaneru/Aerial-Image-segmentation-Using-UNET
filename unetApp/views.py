import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from keras.models import load_model
from django.shortcuts import render
from .serializers import ImageSerializer
from .models import Images
from rest_framework import viewsets, status
from django.http import HttpResponse
from rest_framework.response import Response
import numpy as np
from keras.utils import load_img, img_to_array
from django.conf import settings
import os
import io
from PIL import Image
from django.core.files.base import ContentFile
from django.urls import reverse


os.environ['SM_FRAMEWORK'] = 'tf.keras'
import segmentation_models as sm


class ImageViewSet(viewsets.ModelViewSet):
    queryset = Images.objects.all()
    serializer_class = ImageSerializer


    def preprocess_image(self, uploaded_image):
        # Open the uploaded image using PIL
        pil_image = Image.open(uploaded_image)
        # Resize the image to the desired size
        resized_image = pil_image.resize((256, 256))
        # Convert the PIL image to a numpy array
        image = img_to_array(resized_image) / 255.0
        
        image = image[:, :, :3]
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image

    def mask2img(self, mask):
        palette = {
            0: (226, 169, 41),
            1: (132, 41, 246),
            2: (110, 193, 228),
            3: (60, 16, 152),
            4: (254, 221, 58),
            5: (155, 155, 155)
        }
        rows = mask.shape[1]
        cols = mask.shape[2]
        image = np.zeros((rows, cols, 3), dtype=np.uint8)
        for j in range(rows):
            for i in range(cols):
                class_index = np.argmax(mask[0, j, i])
                if class_index in palette:
                    image[j, i] = palette[class_index]
                else:
                    # Handle the case when the class index is not in the palette
                    # For example, set the pixel to black or any other color
                    image[j, i] = (0, 0, 0)  # Black color
        return image
    
    def calculate_area_percentage(self, mask_image):
        class_color_map = {
            "Building": [60, 16, 152],
            "Land": [132, 41, 246],
            "Road": [110, 193, 228],
            "Vegetation": [254, 221, 58],
            "Water": [226, 169, 41],
            "Unlabeled": [155, 155, 155]
        }
        class_areas = {}

        if mask_image.size == 0:
            # Handle empty mask_image
            return class_areas

        mask_array = np.array(mask_image)
        if len(mask_array.shape) != 3:
            # Ensure mask_image has expected dimensions
            return class_areas

        total_pixels = mask_array.shape[0] * mask_array.shape[1]
        for class_name, color in class_color_map.items():
            class_pixels = np.sum(np.all(mask_array == color, axis=2))
            class_area_percentage = (class_pixels / total_pixels) * 100
            class_areas[class_name] = class_area_percentage
        return class_areas

    def create(self, request, *args, **kwargs):

        uploaded_image = request.FILES.get('unMaskedImage')

        # Preprocess the image
        preprocessed_image = self.preprocess_image(uploaded_image)
        
        # Define RGB values for each class
        class_building = np.array([60, 16, 152])  # RGB values for building
        class_land = np.array([132, 41, 246])      # RGB values for land
        class_road = np.array([110, 193, 228])     # RGB values for road
        class_vegetation = np.array([254, 221, 58]) # RGB values for vegetation
        class_water = np.array([226, 169, 41])      # RGB values for water
        class_unlabeled = np.array([155, 155, 155]) # RGB values for unlabeled

    
        def jaccard_coef(y_true, y_pred):
            y_true_flatten = K.flatten(y_true)
            y_pred_flatten = K.flatten(y_pred)
            intersection = K.sum(y_true_flatten * y_pred_flatten)
            final_coef_value = (intersection + 1.0) / (K.sum(y_true_flatten) +
                                                       K.sum(y_pred_flatten) - intersection + 1.0)
            return final_coef_value

        weights = [0.1666, 0.1666, 0.1666, 0.1666, 0.1666, 0.1666]

        # Define custom loss objects
        class DiceLoss(sm.losses.DiceLoss):
            def __init__(self, class_weights=None):
                super().__init__(class_weights=class_weights)

        class CategoricalFocalLoss(sm.losses.CategoricalFocalLoss):
            def __init__(self):
                super().__init__()

        def total_loss(y_true, y_pred):
            return DiceLoss(class_weights=weights)(y_true, y_pred) + CategoricalFocalLoss()(y_true, y_pred)

        # Define the custom loss function 'dice_loss_plus_1focal_loss'
        def dice_loss_plus_1focal_loss(y_true, y_pred):
            return DiceLoss(class_weights=weights)(y_true, y_pred) + CategoricalFocalLoss()(y_true, y_pred)

        # Define custom objects dictionary with the custom loss function
        custom_objects = {
            'jaccard_coef': jaccard_coef,
            'total_loss': total_loss,
            'DiceLoss': DiceLoss,
            'CategoricalFocalLoss': CategoricalFocalLoss,
            'dice_loss_plus_1focal_loss': dice_loss_plus_1focal_loss
        }

        model_file_path = os.path.join(settings.BASE_DIR, 'segment_models', 'next25epochcrossvalidation.h5')
        
        metrics = ["accuracy", jaccard_coef]
        
        # Load model
        model = load_model(model_file_path,
                           custom_objects=custom_objects, compile=False)
        
        model.compile(optimizer="adam", loss=total_loss, metrics=metrics)
        

        # Perform prediction
        with tf.device('/cpu:0'):  # Use CPU for prediction
            predicted_mask = model.predict(preprocessed_image)

        # Save the uploaded image to the unMasked directory
        unmasked_image_path = os.path.join(
            settings.MEDIA_ROOT, 'images', 'unMasked', uploaded_image.name)
        with open(unmasked_image_path, 'wb') as f:
            f.write(uploaded_image.read())

        # Postprocess the predicted mask
        predicted_mask = self.mask2img(predicted_mask)
        # Calculate area percentage of each class in the predicted mask
        class_areas = self.calculate_area_percentage(predicted_mask)
        # Save the predicted mask image to the masked directory
        # Assuming predicted_mask is a numpy array representing the image
        masked_image_path = os.path.join(
            settings.MEDIA_ROOT, 'images', 'masked', uploaded_image.name)
        # Convert the predicted_mask numpy array to an image file
        # and save it to the masked directory

        predicted_mask = Image.fromarray(predicted_mask)

        # Save the PIL Image object to the specified path
        predicted_mask.save(masked_image_path)

        # Save the predicted mask file path to the database
        instance = Images.objects.create(unMaskedImage=uploaded_image, predictedMask=masked_image_path)
        
        # Serialize the instance and include the predicted mask URL in the response
        serializer = self.get_serializer(instance)
        response_data = serializer.data
        response_data = {
            'predictedMaskUrl': request.build_absolute_uri(instance.predictedMask.url),
            'classAreas': class_areas
        }

        return Response(response_data, status=status.HTTP_201_CREATED)