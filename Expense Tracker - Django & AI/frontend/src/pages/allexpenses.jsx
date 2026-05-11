import ExpenseList
from "../components/ExpenseList";

import { useEffect, useState }
from "react";

import api
from "../services/api";


function AllExpenses() {

    const [expenses,
        setExpenses] =
        useState([]);


    const fetchExpenses =
        async () => {

        try {

            const response =
                await api.get(
                    "/api/expenses/"
                );

            setExpenses(
                response.data
            );

        } catch (error) {

            console.error(error);
        }
    };


    useEffect(() => {

        fetchExpenses();

    }, []);


    return (

        <div className="min-h-screen bg-slate-950 p-8">

            <div className="max-w-7xl mx-auto">

                <div className="mb-10">

                    <h1 className="text-5xl font-bold text-white">

                        All Expenses

                    </h1>

                    <p className="text-slate-400 mt-3">

                        Complete expense history

                    </p>

                </div>


                <div className="bg-slate-900 border border-slate-800 rounded-3xl p-8">

                    <ExpenseList
                        expenses={expenses}
                        fetchExpenses={
                            fetchExpenses
                        }
                    />

                </div>

            </div>

        </div>
    );
}

export default AllExpenses;