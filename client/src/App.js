import React from 'react'; // Import React library
import { Route, Routes } from 'react-router-dom'; // Import Route and Routes from react-router-dom for routing
import StudentRoute from './route/StudentRoute'; // Import a custom route component

const App = () => {
  return (
    <div className='app'> {/* Main container for the application */}
      <Routes>
        {/* Define routes for the application */}
        <Route path='*' element={<StudentRoute />} />
        {/* Route wildcard '*' is used to match any path */}
        {/* StudentRoute component will handle routing logic */}
      </Routes>
    </div>
  );
}

export default App; // Export the App component for use in other parts of the application
