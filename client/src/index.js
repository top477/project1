import React from 'react'; // Import the React library
import ReactDOM from 'react-dom/client'; // Import ReactDOM for rendering the React application
import './index.css'; // Import the global CSS styles
import App from './App'; // Import the main App component
import reportWebVitals from './reportWebVitals'; // Import the function to measure performance metrics
import { BrowserRouter } from 'react-router-dom'; // Import BrowserRouter for handling routing

// Create a root element for rendering the React application
const root = ReactDOM.createRoot(document.getElementById('root'));

// Render the React application inside the root element
root.render(
  <React.StrictMode>
    {/* StrictMode helps to identify potential problems in the application during development */}
    <BrowserRouter>
      {/* BrowserRouter enables client-side routing for the application */}
      <App />
      {/* The App component is the main component that will be rendered */}
    </BrowserRouter>
  </React.StrictMode>
);

// Measure performance metrics of the application
reportWebVitals();
