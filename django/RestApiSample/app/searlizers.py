from rest_framework import serializers
from app.models import Article, User

class UserSerializer(serializers.ModelSerializer):
  class Meta:
    model = User

class ArticleSerializer(serializers.ModelSerializer):
  class Meta:
    model = Article
