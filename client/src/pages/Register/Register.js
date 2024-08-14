import React, { useState } from 'react';
import axios from 'axios';
import './style.css';
import Button from '../../components/Button/Button';
import { useNavigate } from 'react-router-dom';
import config from '../../config/Apiconfig';

const Register = () => {
    // State variables to hold form input values
    const [name, setName] = useState('');
    const [username, setUserName] = useState('');
    const [password, setPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [phone, setPhone] = useState('');
    const [age, setAge] = useState('');

    // Hook for navigation
    const navigate = useNavigate();

    // Function to handle form submission
    const handleSubmit = async(e) => {
        e.preventDefault();

        // Check if password and confirm password match
        if (password !== confirmPassword) {
            alert("Passwords don't match");
            return;
        }

        // Create a user object with the form data
        const user = {
            name,
            username,
            password,
            phone,
            age,
        };

        try {
            // Send a POST request to register the user
            const response = await axios.post(`${config.baseURL}`, user);
            console.log('response', response.data);

            // If registration is successful, clear the form and navigate to the login page
            if (response.data.success) {
                alert(response.data.message);
                setName('');
                setUserName('');
                setPassword('');
                setConfirmPassword('');
                setPhone('');
                setAge('');
                navigate('/login');
            }
        } catch (error) {
            // Handle errors during the registration process
            console.log('error', error);
            alert(error.response.data.message);
        }
    };

    return (
        <div className="register-container">
            <h2>REGISTER</h2>
            {/* Form for user registration */}
            <form onSubmit={handleSubmit}>
                <div className="form-group">
                    <label>Full Name</label>
                    <input
                        type="text"
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        required
                    />
                </div>
                <div className="form-group">
                    <label>Email</label>
                    <input
                        type="email"
                        value={username}
                        onChange={(e) => setUserName(e.target.value)}
                        required
                    />
                </div>
                <div className="form-group-row">
                    <div className="form-group half-width">
                        <label>Phone</label>
                        <input
                            type="text"
                            value={phone}
                            onChange={(e) => setPhone(e.target.value)}
                            required
                        />
                    </div>
                    <div className="form-group half-width">
                        <label>Age</label>
                        <input
                            type="number"
                            value={age}
                            onChange={(e) => setAge(e.target.value)}
                            required
                        />
                    </div>
                </div>
                <div className="form-group">
                    <label>Password</label>
                    <input
                        type="password"
                        value={password}
                        onChange={(e) => setPassword(e.target.value)}
                        required
                    />
                </div>
                <div className="form-group">
                    <label>Confirm Password</label>
                    <input
                        type="password"
                        value={confirmPassword}
                        onChange={(e) => setConfirmPassword(e.target.value)}
                        required
                    />
                </div>
                {/* Submit button */}
                <Button type="submit" text="Register" />
            </form>
        </div>
    );
};

export default Register;
