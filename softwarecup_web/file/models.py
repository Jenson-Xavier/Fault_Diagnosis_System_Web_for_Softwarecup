from django.db import models
from django.utils import timezone

# Create your models here.

# 文件的名称和路径存放到数据库模型中


class FileModel(models.Model):
    name = models.CharField(max_length=100)
    path = models.CharField(max_length=200)
    upload_time = models.DateField(default=timezone.now)
