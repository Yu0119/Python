from rest_framework import viewsets, routers
from app.models import Article
from app.serializers import ArticleSerializer

class ArticleViewSet(viewsets.ModelViewSet):
  queryset = Article.objects.all()
  serializer_class = ArticleSerializer

router = routers.DefaultRouter()
router.register(r'articles', ArticleViewSet)
