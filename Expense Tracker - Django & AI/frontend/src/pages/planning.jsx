import { useState, useEffect } from "react";
import BudgetSection from "../components/budgetsection";
import SavingsGoals from "../components/savingsgoals";
import api from "../services/api";

function Planning() {

    const [budgets, setBudgets] = useState([]);
    const [savingsGoals, setSavingsGoals] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        fetchData();
    }, []);

    const fetchData = async () => {
        setLoading(true);
        setError(null);
        try {
            await Promise.all([fetchBudgets(), fetchSavingsGoals()]);
        } catch (err) {
            setError("Failed to load planning data. Please check your connection and try again.");
            console.error("Error fetching planning data:", err);
        } finally {
            setLoading(false);
        }
    };

    const fetchBudgets = async () => {
        try {
            const response = await api.get("/api/budgets/");
            setBudgets(response.data);
        } catch (error) {
            console.error("Error fetching budgets:", error);
            throw error;
        }
    };

    const fetchSavingsGoals = async () => {
        try {
            const response = await api.get("/api/savings-goals/");
            setSavingsGoals(response.data);
        } catch (error) {
            console.error("Error fetching savings goals:", error);
            throw error;
        }
    };

    if (loading) {
        return (
            <div className="min-h-screen bg-slate-950 p-8 flex items-center justify-center">
                <div className="text-white text-xl">Loading planning data...</div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="min-h-screen bg-slate-950 p-8 flex items-center justify-center">
                <div className="text-center">
                    <div className="text-red-400 text-xl mb-4">{error}</div>
                    <button 
                        onClick={fetchData}
                        className="bg-blue-600 hover:bg-blue-700 px-6 py-3 rounded-2xl text-white font-semibold"
                    >
                        Retry
                    </button>
                </div>
            </div>
        );
    }

    return (

        <div className="min-h-screen bg-slate-950 p-8 space-y-10">
            <SavingsGoals goals={savingsGoals} fetchGoals={fetchSavingsGoals} />

            <BudgetSection budgets={budgets} fetchBudgets={fetchBudgets} />

            

        </div>
    );
}

export default Planning;