from django.urls import path

from .views import ParseExpenseView

urlpatterns = [
    path('parse-expense/', ParseExpenseView.as_view()),
]