import { useState } from "react";

import {
    Pencil,
    Trash2,
    CalendarDays,
    Clock3
} from "lucide-react";

import api from "../services/api";


function ExpenseList({
    expenses,
    fetchExpenses
}) {

    const [editingId,
        setEditingId] =
        useState(null);

    const [editedDescription,
        setEditedDescription] =
        useState("");

    const [editedDate,
        setEditedDate] =
        useState("");

    const [editedTime,
        setEditedTime] =
        useState("");

    const [searchQuery,
    setSearchQuery] =
    useState("");

    const [selectedCategory,
        setSelectedCategory] =
        useState("All");

    // DELETE

    const handleDelete =
        async (id) => {

        try {

            const token =
                localStorage.getItem(
                    "token"
                );

            await api.delete(
                `/api/expenses/${id}/`,
                {
                    headers: {
                        Authorization:
                            `Token ${token}`
                    }
                }
            );

            fetchExpenses();

        } catch (error) {

            console.error(error);
        }
    };


    // START EDIT

    const startEditing =
        (expense) => {

        setEditingId(expense.id);

        setEditedDescription(
            expense.description
        );

        const dateObj =
            new Date(expense.date);

        // DATE

        setEditedDate(
            dateObj
                .toISOString()
                .split("T")[0]
        );

        // TIME

        setEditedTime(
            dateObj
                .toTimeString()
                .slice(0, 5)
        );
    };


    // UPDATE

    const handleUpdate =
        async (expense) => {

        try {

            const token =
                localStorage.getItem(
                    "token"
                );

            // PARSE TEXT ONLY

            const parseResponse =
                await api.post(
                    "/ai/parse-expense/",
                    {
                        text:
                            editedDescription,

                        save: false
                    },
                    {
                        headers: {
                            Authorization:
                                `Token ${token}`
                        }
                    }
                );


            const parsedExpense =
                parseResponse.data.expenses
                ? parseResponse
                    .data
                    .expenses[0]
                : parseResponse.data;


            // COMBINE DATE + TIME

            const combinedDateTime =
              `${editedDate}T${editedTime}:00`;


            // PATCH UPDATE

            await api.patch(
                `/api/expenses/${expense.id}/`,
                {
                    description:
                        editedDescription,

                    amount:
                        parsedExpense.amount,

                    date:
                        combinedDateTime
                },
                {
                    headers: {
                        Authorization:
                            `Token ${token}`
                    }
                }
            );


            setEditingId(null);

            fetchExpenses();

        } catch (error) {

            console.error(error);

            alert(
                "Failed to update expense"
            );
        }
    };

    const filteredExpenses =
        expenses.filter((expense) => {

        const matchesSearch =

            expense.description
            .toLowerCase()
            .includes(
                searchQuery.toLowerCase()
            );


        const matchesCategory =

            selectedCategory === "All"
            ||

            expense.category_name
            === selectedCategory;


        return (
            matchesSearch
            && matchesCategory
        );
    });


    return (

        <div>

            <h2 className="text-2xl font-semibold mb-6">
                Recent Expenses
            </h2>
            <div className="flex flex-col md:flex-row gap-4 mb-6">

    {/* SEARCH */}

    <input
        type="text"
        placeholder="Search expenses..."
        value={searchQuery}
        onChange={(e) =>
            setSearchQuery(
                e.target.value
            )
        }
        className="flex-1 bg-slate-800 border border-slate-700 rounded-2xl px-5 py-3 text-white outline-none"
    />


    {/* CATEGORY FILTER */}

    <select
        value={selectedCategory}
        onChange={(e) =>
            setSelectedCategory(
                e.target.value
            )
        }
        className="bg-slate-800 border border-slate-700 rounded-2xl px-5 py-3 text-white outline-none"
    >

        <option value="All">
            All Categories
        </option>

        <option value="Food">
            Food
        </option>

        <option value="Transportation">
            Transportation
        </option>

        <option value="Shopping">
            Shopping
        </option>

        <option value="Entertainment">
            Entertainment
        </option>

        <option value="Bills">
            Bills
        </option>

        <option value="Healthcare">
            Healthcare
        </option>

        <option value="Education">
            Education
        </option>

        <option value="Travel">
            Travel
        </option>

        <option value="Rent">
            Rent
        </option>

        <option value="Miscellaneous">
            Miscellaneous
        </option>

    </select>

</div>

            <div className="space-y-5">

                {filteredExpenses.map(
                    (expense) => (

                    <div
                        key={expense.id}
                        className="bg-slate-800 border border-slate-700 rounded-2xl p-6"
                    >

                        {editingId
                            === expense.id ? (

                            <div>

                                {/* DESCRIPTION */}

                                <textarea
                                    rows="3"
                                    value={
                                        editedDescription
                                    }
                                    onChange={(e) =>
                                        setEditedDescription(
                                            e.target.value
                                        )
                                    }
                                    className="w-full bg-slate-900 border border-slate-700 rounded-xl p-4 text-white outline-none"
                                />


                                {/* DATE + TIME */}

                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">

                                    <div>

                                        <label className="text-slate-400 text-sm mb-2 block">
                                            Date
                                        </label>

                                        <input
                                            type="date"
                                            value={
                                                editedDate
                                            }
                                            onChange={(e) =>
                                                setEditedDate(
                                                    e.target.value
                                                )
                                            }
                                            className="w-full bg-slate-900 border border-slate-700 rounded-xl px-4 py-3 text-white outline-none"
                                        />

                                    </div>


                                    <div>

                                        <label className="text-slate-400 text-sm mb-2 block">
                                            Time
                                        </label>

                                        <input
                                            type="time"
                                            value={
                                                editedTime
                                            }
                                            onChange={(e) =>
                                                setEditedTime(
                                                    e.target.value
                                                )
                                            }
                                            className="w-full bg-slate-900 border border-slate-700 rounded-xl px-4 py-3 text-white outline-none"
                                        />

                                    </div>

                                </div>


                                {/* SAVE */}

                                <button
                                    onClick={() =>
                                        handleUpdate(
                                            expense
                                        )
                                    }
                                    className="mt-5 bg-blue-600 hover:bg-blue-700 px-5 py-3 rounded-xl font-semibold"
                                >
                                    Save Changes
                                </button>

                            </div>

                        ) : (

                            <div>

                                {/* HEADER */}

                                <div className="flex items-start justify-between">

                                    <div>

                                        <h3 className="text-3xl font-bold">
                                            ₹{
                                                Number(
                                                    expense.amount
                                                )
                                                .toLocaleString()
                                            }
                                        </h3>


                                        <p className="text-slate-300 mt-3 text-lg">
                                            {
                                                expense.description
                                            }
                                        </p>


                                        {/* DATE */}

                                        <div className="flex items-center gap-2 text-slate-500 text-sm mt-4">

                                            <CalendarDays
                                                size={15}
                                            />

                                            <span>

                                                {new Date(
                                                    expense.date.replace("Z", "")
                                                )
                                                .toLocaleDateString(
                                                    "en-IN",
                                                    {
                                                        day:
                                                            "numeric",
                                                        month:
                                                            "short",
                                                        year:
                                                            "numeric"
                                                    }
                                                )}

                                            </span>

                                        </div>


                                        {/* TIME */}

                                        <div className="flex items-center gap-2 text-slate-500 text-sm mt-2">

                                            <Clock3
                                                size={15}
                                            />

                                            <span>

                                                {new Date(
                                                    expense.date
                                                )
                                                .toLocaleTimeString(
                                                    "en-IN",
                                                    {
                                                        hour:
                                                            "numeric",
                                                        minute:
                                                            "2-digit"
                                                    }
                                                )}

                                            </span>

                                        </div>

                                    </div>


                                    {/* CATEGORY */}

                                    <span className="bg-blue-500/10 text-blue-400 px-4 py-2 rounded-xl text-sm">

                                        {
                                            expense.category_name
                                        }

                                    </span>

                                </div>


                                {/* ACTIONS */}

                                <div className="flex gap-3 mt-6">

                                    <button
                                        onClick={() =>
                                            startEditing(
                                                expense
                                            )
                                        }
                                        className="flex items-center gap-2 bg-slate-700 hover:bg-slate-600 px-4 py-2 rounded-xl"
                                    >

                                        <Pencil
                                            size={16}
                                        />

                                        Edit

                                    </button>


                                    <button
                                        onClick={() =>
                                            handleDelete(
                                                expense.id
                                            )
                                        }
                                        className="flex items-center gap-2 bg-red-500/10 text-red-400 hover:bg-red-500/20 px-4 py-2 rounded-xl"
                                    >

                                        <Trash2
                                            size={16}
                                        />

                                        Delete

                                    </button>

                                </div>

                            </div>

                        )}

                    </div>

                ))}

            </div>

        </div>
    );
}

export default ExpenseList;