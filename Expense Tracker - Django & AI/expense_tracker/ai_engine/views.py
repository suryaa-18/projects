from datetime import date

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from django.utils import timezone
from expenses.models import Expense, Category
from expenses.serializers import ExpenseSerializer

from .services.parser import JointExpenseParser


class ParseExpenseView(APIView):

    permission_classes = [IsAuthenticated]

    def post(self, request):

        try:

            text = request.data.get("text")
            save_expense = request.data.get(
                "save",
                True
            )
            if not text:

                return Response(
                    {"error": "Text is required"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            parser = JointExpenseParser()

            parsed_expenses = parser.parse(text)
            # ONLY PARSE (NO SAVE)

            if not save_expense:

                return Response({
                    "expenses": [
                        {
                            "amount": item.amount,
                            "category": item.category,
                            "description": item.raw_segment,
                            "confidence": item.confidence
                        }
                        for item in parsed_expenses
                    ]
                })

            created_expenses = []

            for item in parsed_expenses:

                category_name = item.category
                amount = item.amount
                description = item.raw_segment

                category, _ = Category.objects.get_or_create(
                    name=category_name
                )

                expense = Expense.objects.create(
                    user=request.user,
                    amount=amount,
                    category=category,
                    description=description,
                    date=item.datetime,
                    source="nlp",
                    raw_input=text,
                    ai_confidence=item.confidence
                )

                created_expenses.append(expense)

            serializer = ExpenseSerializer(
                created_expenses,
                many=True
            )

            return Response(
                {
                    "expenses": serializer.data,
                    "count": len(created_expenses),
                    "total_amount": sum(
                        e.amount for e in created_expenses
                    )
                }
            )

        except Exception as e:

            return Response(
                {"error": str(e)},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )