from django.urls import path
from .views import *

urlpatterns = [
    path('', index_view, name='index'),
    path('train/', train_view, name='train'),
    path('train/training/<int:id>/', training_view, name='training'),
    path('train/training/<int:id>/train_model_download/',
         train_model_download_view, name='train_download'),
    path('train/downloading/', train_downloading_view, name='train_downloading'),
    path('test/', test_view, name='test'),
    path('test/testing/<int:id>/', testing_view, name='testing'),
    path('test/testing/<int:id>/test_results_download',
         test_download_view, name='test_download'),
    path('test/downloading/', test_downloading_view, name='test_downloading'),
    path('test/visualization', test_visualization_view, name='test_visualization'),
]
