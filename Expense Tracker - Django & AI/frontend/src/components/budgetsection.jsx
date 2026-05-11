import { useState } from "react";

import api from "../services/api";

import {
    Wallet,
    Plus,
    Pencil,
    Trash2
} from "lucide-react";


function BudgetSection({

    budgets,
    fetchBudgets

}) {

    const [categoryName,
        setCategoryName] =
        useState("");

    const [amount,
        setAmount] =
        useState("");

    const [editingBudget,
        setEditingBudget] =
        useState(null);


    // =========================
    // ADD / UPDATE
    // =========================

    const handleAddBudget =
        async () => {

        if (
            !categoryName.trim()
            || !amount
        ) {

            alert(
                "Please enter category and budget."
            );

            return;
        }

        try {

            // UPDATE

            if (editingBudget) {

                await api.put(
                    `/api/budgets/${editingBudget}/`,
                    {
                        category:
                            categoryName,

                        monthly_limit:
                            amount
                    }
                );

            }

            // CREATE

            else {

                await api.post(
                    "/api/budgets/",
                    {
                        category:
                            categoryName,

                        monthly_limit:
                            amount
                    }
                );
            }


            // RESET

            setCategoryName("");

            setAmount("");

            setEditingBudget(null);

            fetchBudgets();

        } catch (error) {

            console.error(error);

            alert(
                "Failed to save budget"
            );
        }
    };


    // =========================
    // EDIT
    // =========================

    const handleEditBudget =
        (budget) => {

        setEditingBudget(
            budget.id
        );

        setCategoryName(
            budget.category_name
        );

        setAmount(
            budget.monthly_limit
        );
    };


    // =========================
    // DELETE
    // =========================

    const handleDeleteBudget =
        async (id) => {

        try {

            await api.delete(
                `/api/budgets/${id}/`
            );

            fetchBudgets();

        } catch (error) {

            console.error(error);

            alert(
                "Failed to delete budget"
            );
        }
    };


    return (

        <div className="bg-slate-900 border border-slate-800 rounded-[32px] p-8">

            {/* HEADER */}

            <div className="flex items-center justify-between mb-8">

                <div>

                    <h2 className="text-3xl font-bold text-white">

                        Budget Management

                    </h2>

                    <p className="text-slate-400 mt-2">

                        Set monthly spending limits

                    </p>

                </div>


                <div className="bg-blue-500/10 border border-blue-500/20 p-4 rounded-2xl">

                    <Wallet
                        className="text-blue-400"
                        size={28}
                    />

                </div>

            </div>


            {/* FORM */}

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">

                {/* CATEGORY */}

                <input
                    type="text"
                    placeholder="Category Name"
                    value={categoryName}
                    onChange={(e) =>
                        setCategoryName(
                            e.target.value
                        )
                    }
                    className="bg-slate-800 border border-slate-700 rounded-2xl px-5 py-4 text-white outline-none focus:border-blue-500"
                />


                {/* AMOUNT */}

                <input
                    type="number"
                    placeholder="Monthly Budget"
                    value={amount}
                    onChange={(e) =>
                        setAmount(
                            e.target.value
                        )
                    }
                    className="bg-slate-800 border border-slate-700 rounded-2xl px-5 py-4 text-white outline-none focus:border-blue-500"
                />


                {/* BUTTON */}

                <button
                    onClick={
                        handleAddBudget
                    }
                    className="bg-blue-600 hover:bg-blue-700 transition-all rounded-2xl text-white font-semibold flex items-center justify-center gap-2"
                >

                    <Plus size={18} />

                    {
                        editingBudget
                            ? "Save Changes"
                            : "Add Budget"
                    }

                </button>

            </div>


            {/* BUDGET LIST */}

            <div className="space-y-5">

                {budgets?.length > 0 ? (

                    budgets.map((budget) => (

                        <div
                            key={budget.id}
                            className="bg-slate-800 border border-slate-700 rounded-3xl p-6"
                        >

                            <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-6">

                                {/* LEFT */}

                                <div>

                                    <h3 className="text-3xl font-bold text-white">

                                        {
                                            budget.category_name
                                            || "Unknown"
                                        }

                                    </h3>

                                    <p className="text-slate-400 mt-3 text-lg">

                                        Monthly Limit

                                    </p>

                                </div>


                                {/* RIGHT */}

                                <div className="flex items-center gap-4">

                                    <h3 className="text-4xl font-bold text-white mr-4">

                                        ₹{
                                            Number(
                                                budget.monthly_limit || 0
                                            ).toLocaleString()
                                        }

                                    </h3>


                                    {/* EDIT */}

                                    <button
                                        onClick={() =>
                                            handleEditBudget(
                                                budget
                                            )
                                        }
                                        className="bg-blue-500/10 hover:bg-blue-500/20 border border-blue-500/20 transition-all rounded-2xl px-5 py-3 text-blue-400 font-semibold flex items-center gap-2"
                                    >

                                        <Pencil size={18} />

                                        Edit

                                    </button>


                                    {/* DELETE */}

                                    <button
                                        onClick={() =>
                                            handleDeleteBudget(
                                                budget.id
                                            )
                                        }
                                        className="bg-red-500/10 hover:bg-red-500/20 border border-red-500/20 transition-all rounded-2xl px-5 py-3 text-red-400 font-semibold flex items-center gap-2"
                                    >

                                        <Trash2 size={18} />

                                        Delete

                                    </button>

                                </div>

                            </div>

                        </div>
                    ))

                ) : (

                    <div className="bg-slate-800 border border-slate-700 rounded-3xl p-10 text-center">

                        <p className="text-slate-400 text-lg">

                            No budgets added yet.

                        </p>

                    </div>
                )}

            </div>

        </div>
    );
}

export default BudgetSection;