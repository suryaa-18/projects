from rest_framework import viewsets
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework import generics
from ai_engine.services.insights import ExpenseInsightsEngine
from .services.financial_health import (
    FinancialHealthEngine
)
from .models import Expense, Category, Budget, SavingsGoal
from .serializers import ExpenseSerializer, CategorySerializer
from .serializers import (
    ExpenseSerializer,
    CategorySerializer,
    BudgetSerializer,
    SavingsGoalSerializer
)

class ExpenseInsightsView(APIView):

    permission_classes = [
        IsAuthenticated
    ]


    def get(self, request):

        data = (
            ExpenseInsightsEngine
            .generate_insights(
                request.user
            )
        )

        return Response(data)

class ExpenseViewSet(viewsets.ModelViewSet):

    serializer_class = ExpenseSerializer

    permission_classes = [IsAuthenticated]

    def get_queryset(self):

        return Expense.objects.filter(
            user=self.request.user
        ).order_by("-date")


class CategoryViewSet(viewsets.ModelViewSet):

    queryset = Category.objects.all()

    serializer_class = CategorySerializer

    permission_classes = [IsAuthenticated]

class BudgetListCreateView(
    generics.ListCreateAPIView
):

    serializer_class = BudgetSerializer

    permission_classes = [
        IsAuthenticated
    ]


    def get_queryset(self):

        return Budget.objects.filter(
            user=self.request.user
        )


    def perform_create(
        self,
        serializer
    ):

        serializer.save(
            user=self.request.user
        )


class BudgetDetailView(
    generics.RetrieveUpdateDestroyAPIView
):

    serializer_class = BudgetSerializer

    permission_classes = [
        IsAuthenticated
    ]


    def get_queryset(self):

        return Budget.objects.filter(
            user=self.request.user
        )
    
class FinancialHealthView(APIView):

    permission_classes = [
        IsAuthenticated
    ]


    def get(self, request):

        data = (
            FinancialHealthEngine
            .calculate(request.user)
        )

        return Response(data)
    
class SavingsGoalViewSet(
    viewsets.ModelViewSet
):

    serializer_class = SavingsGoalSerializer

    permission_classes = [
        IsAuthenticated
    ]


    def get_queryset(self):

        return (
            SavingsGoal.objects.filter(
                user=self.request.user
            )
        )


    def perform_create(
        self,
        serializer
    ):

        serializer.save(
            user=self.request.user
        )