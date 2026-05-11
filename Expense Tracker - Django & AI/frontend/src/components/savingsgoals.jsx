import { useState } from "react";

import api from "../services/api";

import {
    PiggyBank,
    Plus,
    Pencil,
    Trash2,
    Target
} from "lucide-react";


function SavingsGoals({

    goals,
    fetchGoals

}) {

    const [title,
        setTitle] =
        useState("");

    const [targetAmount,
        setTargetAmount] =
        useState("");

    const [currentAmount,
        setCurrentAmount] =
        useState("");

    const [deadline,
        setDeadline] =
        useState("");

    const [editingGoal,
        setEditingGoal] =
        useState(null);


    // =========================
    // ADD / UPDATE
    // =========================

    const handleSaveGoal =
        async () => {

        if (
            !title.trim()
            || !targetAmount
        ) {

            alert(
                "Please enter goal title and target amount."
            );

            return;
        }

        try {

            // UPDATE

            if (editingGoal) {

                await api.put(
                    `/api/savings-goals/${editingGoal}/`,
                    {
                        title,
                        target_amount:
                            targetAmount,

                        current_amount:
                            currentAmount || 0,

                        deadline
                    }
                );
            }

            // CREATE

            else {

                await api.post(
                    "/api/savings-goals/",
                    {
                        title,

                        target_amount:
                            targetAmount,

                        current_amount:
                            currentAmount || 0,

                        deadline
                    }
                );
            }


            // RESET

            setTitle("");

            setTargetAmount("");

            setCurrentAmount("");

            setDeadline("");

            setEditingGoal(null);

            fetchGoals();

        } catch (error) {

            console.error(error);

            alert(
                "Failed to save goal"
            );
        }
    };


    // =========================
    // EDIT
    // =========================

    const handleEditGoal =
        (goal) => {

        setEditingGoal(
            goal.id
        );

        setTitle(
            goal.title
        );

        setTargetAmount(
            goal.target_amount
        );

        setCurrentAmount(
            goal.current_amount
        );

        setDeadline(
            goal.deadline || ""
        );
    };


    // =========================
    // DELETE
    // =========================

    const handleDeleteGoal =
        async (id) => {

        try {

            await api.delete(
                `/api/savings-goals/${id}/`
            );

            fetchGoals();

        } catch (error) {

            console.error(error);

            alert(
                "Failed to delete goal"
            );
        }
    };


    return (

        <div className="bg-slate-900 border border-slate-800 rounded-[32px] p-8">

            {/* HEADER */}

            <div className="flex items-center justify-between mb-8">

                <div>

                    <h2 className="text-3xl font-bold text-white">

                        Savings Goals

                    </h2>

                    <p className="text-slate-400 mt-2">

                        Track your financial targets

                    </p>

                </div>


                <div className="bg-emerald-500/10 border border-emerald-500/20 p-4 rounded-2xl">

                    <PiggyBank
                        className="text-emerald-400"
                        size={28}
                    />

                </div>

            </div>


            {/* FORM */}

            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4 mb-8">

                {/* TITLE */}

                <input
                    type="text"
                    placeholder="Goal Name"
                    value={title}
                    onChange={(e) =>
                        setTitle(
                            e.target.value
                        )
                    }
                    className="bg-slate-800 border border-slate-700 rounded-2xl px-5 py-4 text-white outline-none focus:border-emerald-500"
                />


                {/* TARGET */}

                <input
                    type="number"
                    placeholder="Target Amount"
                    value={targetAmount}
                    onChange={(e) =>
                        setTargetAmount(
                            e.target.value
                        )
                    }
                    className="bg-slate-800 border border-slate-700 rounded-2xl px-5 py-4 text-white outline-none focus:border-emerald-500"
                />


                {/* CURRENT */}

                <input
                    type="number"
                    placeholder="Current Savings"
                    value={currentAmount}
                    onChange={(e) =>
                        setCurrentAmount(
                            e.target.value
                        )
                    }
                    className="bg-slate-800 border border-slate-700 rounded-2xl px-5 py-4 text-white outline-none focus:border-emerald-500"
                />


                {/* DEADLINE */}

                <input
                    type="date"
                    value={deadline}
                    onChange={(e) =>
                        setDeadline(
                            e.target.value
                        )
                    }
                    className="bg-slate-800 border border-slate-700 rounded-2xl px-5 py-4 text-white outline-none focus:border-emerald-500"
                />

            </div>


            {/* BUTTON */}

            <button
                onClick={
                    handleSaveGoal
                }
                className="mb-10 bg-emerald-600 hover:bg-emerald-700 transition-all rounded-2xl px-8 py-4 text-white font-semibold flex items-center gap-3"
            >

                <Plus size={18} />

                {
                    editingGoal
                        ? "Save Changes"
                        : "Add Goal"
                }

            </button>


            {/* GOALS */}

            <div className="space-y-6">

                {goals?.length > 0 ? (

                    goals.map((goal) => {

                        const progress =
                            (
                                Number(goal.current_amount)
                                /
                                Number(goal.target_amount)
                            ) * 100;

                        return (

                            <div
                                key={goal.id}
                                className="bg-slate-800 border border-slate-700 rounded-3xl p-6"
                            >

                                <div className="flex flex-col xl:flex-row xl:items-center xl:justify-between gap-6">

                                    {/* LEFT */}

                                    <div className="flex-1">

                                        <div className="flex items-center gap-3 mb-4">

                                            <Target
                                                className="text-emerald-400"
                                                size={24}
                                            />

                                            <h3 className="text-3xl font-bold text-white">

                                                {goal.title}

                                            </h3>

                                        </div>


                                        {/* PROGRESS BAR */}

                                        <div className="w-full bg-slate-700 rounded-full h-4 overflow-hidden mb-4">

                                            <div
                                                className="bg-emerald-500 h-full rounded-full transition-all"
                                                style={{
                                                    width:
                                                        `${Math.min(progress, 100)}%`
                                                }}
                                            />

                                        </div>


                                        <div className="flex flex-wrap items-center gap-6 text-lg">

                                            <p className="text-slate-300">

                                                ₹{
                                                    Number(
                                                        goal.current_amount
                                                    ).toLocaleString()
                                                }

                                            </p>

                                            <p className="text-slate-500">

                                                of

                                            </p>

                                            <p className="text-white font-semibold">

                                                ₹{
                                                    Number(
                                                        goal.target_amount
                                                    ).toLocaleString()
                                                }

                                            </p>

                                            <p className="text-emerald-400 font-semibold">

                                                {progress.toFixed(0)}%

                                            </p>

                                        </div>


                                        {goal.deadline && (

                                            <p className="text-slate-400 mt-4">

                                                Deadline:
                                                {" "}
                                                {
                                                    new Date(
                                                        goal.deadline
                                                    ).toLocaleDateString(
                                                        "en-IN"
                                                    )
                                                }

                                            </p>
                                        )}

                                    </div>


                                    {/* ACTIONS */}

                                    <div className="flex items-center gap-4">

                                        <button
                                            onClick={() =>
                                                handleEditGoal(
                                                    goal
                                                )
                                            }
                                            className="bg-blue-500/10 hover:bg-blue-500/20 border border-blue-500/20 transition-all rounded-2xl px-5 py-3 text-blue-400 font-semibold flex items-center gap-2"
                                        >

                                            <Pencil size={18} />

                                            Edit

                                        </button>


                                        <button
                                            onClick={() =>
                                                handleDeleteGoal(
                                                    goal.id
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
                        );
                    })

                ) : (

                    <div className="bg-slate-800 border border-slate-700 rounded-3xl p-10 text-center">

                        <p className="text-slate-400 text-lg">

                            No savings goals added yet.

                        </p>

                    </div>
                )}

            </div>

        </div>
    );
}

export default SavingsGoals;