from django.db import models


#The upload_path function concatenates the images/unMasked/ directory with the filename. 
#This means that uploaded images will be stored in the images/unMasked/ directory inside your media root directory.
def upload_path(instace, filename):
    return '/'.join(['images','unMasked',filename])


class Images(models.Model):
    unMaskedImage = models.ImageField(blank=True, null=True,upload_to=upload_path) #upload_path function determines the upload path for the images.
    predictedMask = models.ImageField(blank=True, null=True, upload_to='images/masked/')
    
    
#The upload_to parameter in Django's ImageField is optional. 
#If you don't provide it, Django will use a default location based on the app label and model name. 
# However, providing a custom upload_to function allows you to define a more specific location or logic for storing uploaded files.

#In the context of receiving images from the frontend, you typically want to specify the upload_to parameter to control where the images will be stored on your server. 
# This allows you to organize your uploaded files and manage them more effectively.