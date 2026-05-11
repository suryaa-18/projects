from rest_framework import serializers

from .models import (
    Budget,
    Expense,
    Category,
    SavingsGoal
)


# =========================
# CATEGORY
# =========================

class CategorySerializer(
    serializers.ModelSerializer
):

    class Meta:

        model = Category

        fields = "__all__"


# =========================
# EXPENSE
# =========================

class ExpenseSerializer(
    serializers.ModelSerializer
):

    category_name = serializers.CharField(
        source="category.name",
        read_only=True
    )

    category = serializers.CharField(
        write_only=True,
        required=False
    )

    class Meta:

        model = Expense

        fields = "__all__"


    def create(
        self,
        validated_data
    ):

        category_name = validated_data.pop(
            "category",
            None
        )

        if category_name:

            category, _ = (
                Category.objects.get_or_create(
                    name=category_name.title()
                )
            )

            validated_data[
                "category"
            ] = category

        return Expense.objects.create(
            **validated_data
        )


    def update(
        self,
        instance,
        validated_data
    ):

        category_name = validated_data.pop(
            "category",
            None
        )

        if category_name:

            category, _ = (
                Category.objects.get_or_create(
                    name=category_name.title()
                )
            )

            instance.category = category

        for attr, value in validated_data.items():
            setattr(instance, attr, value)

        instance.save()

        return instance


# =========================
# BUDGET
# =========================

class BudgetSerializer(
    serializers.ModelSerializer
):

    category_name = serializers.CharField(
        source="category.name",
        read_only=True
    )

    category = serializers.CharField(
        write_only=True,
        required=False
    )

    class Meta:

        model = Budget

        fields = [
            "id",
            "category",
            "category_name",
            "monthly_limit"
        ]


    def create(
        self,
        validated_data
    ):

        category_name = validated_data.pop(
            "category",
            None
        )

        if category_name:

            category, _ = (
                Category.objects.get_or_create(
                    name=category_name.title()
                )
            )

            validated_data[
                "category"
            ] = category

        return Budget.objects.create(
            **validated_data
        )


    def update(
        self,
        instance,
        validated_data
    ):

        category_name = validated_data.pop(
            "category",
            None
        )

        if category_name:

            category, _ = (
                Category.objects.get_or_create(
                    name=category_name.title()
                )
            )

            instance.category = category


        instance.monthly_limit = (
            validated_data.get(
                "monthly_limit",
                instance.monthly_limit
            )
        )

        instance.save()

        return instance


# =========================
# SAVINGS GOALS
# =========================

class SavingsGoalSerializer(
    serializers.ModelSerializer
):

    class Meta:

        model = SavingsGoal

        fields = [
            "id",
            "title",
            "target_amount",
            "current_amount",
            "deadline",
            "created_at"
        ]