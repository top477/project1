import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Content from '../Content/Content';
import Main from '../Main/Main';
import Welcome from '../Welcome/Welcome';

const Protected = () => {
    const [isAuthenticated, setIsAuthenticated] = useState(false);
    const navigate = useNavigate();

    useEffect(() => {
        const token = localStorage.getItem('token');
        if (token) {
            setIsAuthenticated(true);
        } else {
            setIsAuthenticated(false);
            navigate('/'); 
        }
    }, [navigate]);

    if (isAuthenticated) {
        // return <Content />;
        return <Welcome />
    } else {
        return <Main />;
    }
};

export default Protected;
