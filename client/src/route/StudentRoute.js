import React from 'react';
import Navbar from '../Navbar/Navbar';
import { Route, Routes } from 'react-router-dom';
import Main from '../pages/Main/Main';
import Content from '../pages/Content/Content';
import LogIn from '../pages/LogIn/LogIn';
import Register from '../pages/Register/Register';
import ProtectedRoute from './ProtectedRoute';
import Protected from '../pages/Protected/Protected';
import Certificate from '../components/Certificate/Certificate';
import Welcome from '../pages/Welcome/Welcome'
import Quiz from '../components/Quiz/Quiz'
import SetGoal from '../pages/SetGoal/SetGoal'

const StudentRoute = () => {
  return (
    <div>
      {/* Navbar component to be displayed on all pages */}
      <div>
        <Navbar />
      </div>
      
      {/* Define all the application routes */}
      <Routes>
        {/* Route for the main page */}
        <Route path='/' element={<Main />} />

        {/* Route for the login page */}
        <Route path='/login' element={<LogIn />} />

        {/* Route for the registration page */}
        <Route path='/register' element={<Register />} />

        {/* Route for the protected page (insecure, not using ProtectedRoute) */}
        <Route path='/protected' element={<Protected />} />

        {/* Route for the welcome page (insecure, duplicate path with ProtectedRoute) */}
        <Route path='/protected' element={<Welcome />} />

        {/* Route for the certificate page */}
        <Route path='/certificate' element={<Certificate />} />

        {/* Route for the quiz page with a specific level prop */}
        <Route path='/assessment-quiz' element={<Quiz level='assessment' />} />

        {/* Route for the set goals page */}
        <Route path='/goal' element={<SetGoal />} />

        {/* Protected route for the content page, only accessible if authenticated */}
        <Route 
          path='/content' 
          element={
            <ProtectedRoute>
              <Content />
            </ProtectedRoute>
          }
        />

        {/* Protected route for the welcome page, only accessible if authenticated */}
        <Route 
          path='/welcome' 
          element={
            <ProtectedRoute>
              <Welcome />
            </ProtectedRoute>
          }
        />
      </Routes>
    </div>
  );
};

export default StudentRoute;
