import React from 'react';
import { Navigate } from 'react-router-dom';

const ProtectedRoute = ({ children }) => {
    // Retrieve the token from localStorage
    const token = localStorage.getItem('token');

    // If no token is found, redirect the user to the homepage
    if (!token) {
        return <Navigate to="/" />;
    }

    // If a token is found, render the child components (protected content)
    return children;
};

export default ProtectedRoute;
