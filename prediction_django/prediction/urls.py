from django.urls import path
from .views import LayoutNerAPIView

urlpatterns = [
    path("api/predict", LayoutNerAPIView.as_view(), name="predict"),
]
