from django.db import models

class User(models.Model):
  name = models.CharField(max_length=64)

class Article(models.Model):
  user = models.ForeignKey(User, related_name='article')
  title = models.CharField(max_length=1024)
  contents = models.TextField()
