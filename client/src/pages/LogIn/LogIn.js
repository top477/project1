import React, { useState } from "react";
import "./style.css";
import Button from "../../components/Button/Button";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import config from "../../config/Apiconfig";

const LogIn = () => {
  // State variables to manage the input fields for username and password
  const [username, setUserName] = useState("");
  const [password, setPassword] = useState("");

  // Hook to navigate programmatically between routes
  const navigate = useNavigate();

  // Function to handle form submission and user authentication
  const handleSubmit = async (e) => {
    e.preventDefault(); // Prevent page refresh on form submission
    
    const user = {
      username, // User input for username
      password, // User input for password
    };

    try {
      // Send a POST request to the login API endpoint
      const response = await axios.post(`${config.baseURL}/login`, user);
      
      // Extract the token from the response data
      const token = response.data.data.split(" ")[1];

      // If the login is successful, store the token in localStorage
      if (response.data.success) {
        localStorage.setItem("token", token);

        // Display a success message to the user
        alert(response.data.message);

        // Clear the input fields
        setUserName("");
        setPassword("");

        // Navigate to the protected route after successful login
        navigate("/protected");
      }
    } catch (error) {
      // Log the error to the console
      console.log("error", error);

      // Display an error message to the user
      alert(error.response?.data?.message || "An error occurred");
    }
  };

  return (
    <div className="login-container">
      <h2>LOGIN</h2>
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label>Username</label>
          <input
            type="email"
            value={username}
            placeholder="Enter your email"
            onChange={(e) => setUserName(e.target.value)}
            required
          />
        </div>
        <div className="form-group">
          <label>Password</label>
          <input
            type="password"
            value={password}
            placeholder="Enter your password"
            onChange={(e) => setPassword(e.target.value)}
            required
          />
        </div>
        <Button type="submit" text="Login" />
      </form>
    </div>
  );
};

export default LogIn;
