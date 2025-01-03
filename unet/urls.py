from django.contrib import admin
from django.urls import path, include
from rest_framework import routers
from unetApp.views import ImageViewSet
from django.conf.urls.static import static
from django.conf import settings

router = routers.DefaultRouter() #This router will automatically generate the URL patterns for your ViewSets.
router.register('Images', ImageViewSet)

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include(router.urls))
]

urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
#This is necessary to serve uploaded images from the MEDIA_ROOT directory during development.
