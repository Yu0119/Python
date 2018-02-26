from django.db import models

class User(models.Model):
  name = models.CharField(max_length=64)
  
  def __str__(self):
    return self.name

class Article(models.Model):
  user = models.ForeignKey(User, related_name='article')
  title = models.CharField(max_length=1024)
  contents = models.TextField()
  
  def __str__(self):
    return self.title
