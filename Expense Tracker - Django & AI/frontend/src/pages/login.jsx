import { useState } from "react";

import {
    Link,
    useNavigate
} from "react-router-dom";

import {
    Eye,
    EyeOff,
    BrainCircuit
} from "lucide-react";

import api from "../services/api";


function Login() {

    const navigate = useNavigate();

    const [username,
        setUsername] =
        useState("");

    const [password,
        setPassword] =
        useState("");

    const [loading,
        setLoading] =
        useState(false);

    const [showPassword,
        setShowPassword] =
        useState(false);

    const [error,
        setError] =
        useState("");


    const handleLogin =
        async (e) => {

        e.preventDefault();

        setError("");

        setLoading(true);

        try {

            const response =
                await api.post(
                    "/api/login/",
                    {
                        username,
                        password
                    }
                );


            // SAVE TOKEN

            localStorage.setItem(
                "token",
                response.data.token
            );


            // SAVE USERNAME

            localStorage.setItem(
                "username",
                username
            );


            navigate("/dashboard");

        } catch (error) {

            console.error(error);

            setError(
                "Invalid username or password"
            );

        } finally {

            setLoading(false);
        }
    };


    return (

        <div className="min-h-screen bg-slate-950 flex items-center justify-center px-6 relative overflow-hidden">

            {/* BACKGROUND GLOW */}

            <div className="absolute top-[-120px] left-[-120px] h-72 w-72 bg-blue-500/20 blur-3xl rounded-full" />

            <div className="absolute bottom-[-120px] right-[-120px] h-72 w-72 bg-violet-500/20 blur-3xl rounded-full" />


            {/* LOGIN CARD */}

            <div className="relative w-full max-w-md bg-slate-900/95 backdrop-blur-xl border border-slate-800 rounded-[32px] p-10 shadow-2xl">

                {/* HEADER */}

                <div className="text-center mb-10">

                    <div className="mx-auto bg-blue-500/10 border border-blue-500/20 h-20 w-20 rounded-3xl flex items-center justify-center mb-6">

                        <BrainCircuit
                            className="text-blue-400"
                            size={38}
                        />

                    </div>


                    <h1 className="text-4xl font-bold text-white">
                        Welcome Back
                    </h1>

                    <p className="text-slate-400 mt-3 text-lg">
                        Login to your AI Expense Tracker
                    </p>

                </div>


                {/* ERROR */}

                {error && (

                    <div className="bg-red-500/10 border border-red-500/20 text-red-400 rounded-2xl px-4 py-3 mb-6 text-sm">

                        {error}

                    </div>
                )}


                {/* FORM */}

                <form
                    onSubmit={handleLogin}
                    className="space-y-6"
                >

                    {/* USERNAME */}

                    <div>

                        <label className="block text-slate-300 mb-2 text-sm font-medium">

                            Username

                        </label>

                        <input
                            type="text"
                            placeholder="Enter username"
                            value={username}
                            onChange={(e) =>
                                setUsername(
                                    e.target.value
                                )
                            }
                            className="w-full bg-slate-800 border border-slate-700 rounded-2xl px-5 py-4 text-white outline-none focus:border-blue-500 transition-all"
                            required
                        />

                    </div>


                    {/* PASSWORD */}

                    <div>

                        <label className="block text-slate-300 mb-2 text-sm font-medium">

                            Password

                        </label>


                        <div className="relative">

                            <input
                                type={
                                    showPassword
                                    ? "text"
                                    : "password"
                                }
                                placeholder="Enter password"
                                value={password}
                                onChange={(e) =>
                                    setPassword(
                                        e.target.value
                                    )
                                }
                                className="w-full bg-slate-800 border border-slate-700 rounded-2xl px-5 py-4 text-white outline-none focus:border-blue-500 transition-all pr-14"
                                required
                            />


                            <button
                                type="button"
                                onClick={() =>
                                    setShowPassword(
                                        !showPassword
                                    )
                                }
                                className="absolute right-4 top-1/2 -translate-y-1/2 text-slate-400 hover:text-white"
                            >

                                {showPassword ? (

                                    <EyeOff size={20} />

                                ) : (

                                    <Eye size={20} />

                                )}

                            </button>

                        </div>

                    </div>


                    {/* LOGIN BUTTON */}

                    <button
                        type="submit"
                        disabled={loading}
                        className="w-full mt-4 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 transition-all rounded-2xl py-4 font-semibold text-white text-lg shadow-lg shadow-blue-500/20"
                    >

                        {loading
                            ? "Logging in..."
                            : "Login"}

                    </button>

                </form>


                {/* FOOTER */}

                <p className="text-center text-slate-400 mt-8">

                    Don’t have an account?{" "}

                    <Link
                        to="/signup"
                        className="text-blue-400 hover:text-blue-300 font-medium"
                    >

                        Create Account

                    </Link>

                </p>

            </div>

        </div>
    );
}

export default Login;