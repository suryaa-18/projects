import {
    BrowserRouter,
    Routes,
    Route
} from "react-router-dom";

import Login from "./pages/Login";
import Signup from "./pages/Signup";
import Dashboard from "./pages/Dashboard";
import Planning from "./pages/planning";
import AllExpenses from "./pages/allexpenses";

function App() {

    return (

        <BrowserRouter>

            <Routes>

                <Route
                    path="/"
                    element={<Login />}
                />

                <Route
                    path="/signup"
                    element={<Signup />}
                />

                <Route
                    path="/dashboard"
                    element={<Dashboard />}
                />

                <Route
                    path="/planning"
                    element={<Planning />}
                />

                <Route
                    path="/expenses"
                    element={<AllExpenses />}
                />

            </Routes>

        </BrowserRouter>
    );
}

export default App;