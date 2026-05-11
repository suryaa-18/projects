from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .auth_views import SignupView, LoginView
from .views import BudgetDetailView, BudgetListCreateView, ExpenseInsightsView, FinancialHealthView, SavingsGoalViewSet
from .views import ExpenseViewSet, CategoryViewSet
from .views import FinancialHealthView

router = DefaultRouter()

router.register(
    r'expenses',
    ExpenseViewSet,
    basename='expenses'
)
router.register(
    r'categories',
    CategoryViewSet,
    basename='categories'
)

router.register(
    r'savings-goals',
    SavingsGoalViewSet,
    basename='savings-goals'
)

urlpatterns = [
    path('signup/', SignupView.as_view()),
    path('login/', LoginView.as_view()),
    path('expenses/insights/', ExpenseInsightsView.as_view()),
    path(
        'budgets/',
        BudgetListCreateView.as_view()
    ),
    path(
        'expenses/financial-health/',
        FinancialHealthView.as_view()
    ),
        path(
        'budgets/<int:pk>/',
        BudgetDetailView.as_view()
    ),
    path('', include(router.urls)),
]