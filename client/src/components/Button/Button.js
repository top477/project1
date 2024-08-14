import React from 'react';  // Import the React library
import './style.css';  // Import the CSS file for styling

/**
 * Button Component
 *
 * This is a reusable button component that accepts several props to customize its behavior and style.
 *
 * @param {string} text - The text displayed on the button.
 * @param {function} onClick - The function to be called when the button is clicked.
 * @param {string} backgroundColor - The background color of the button.
 * @param {string} textColor - The text color of the button. Defaults to 'white' if not provided.
 * @param {string} type - The type of the button. Defaults to 'submit'.
 */
const Button = ({ text, onClick, backgroundColor, textColor, type='submit' }) => {
  // Define the style object for the button
  const buttonStyle = {
    backgroundColor: backgroundColor,  // Set the background color from props
    color: textColor || 'white',  // Set the text color from props, default to 'white'
  };

  return (
    // Render the button element with the defined style and event handler
    <button 
      className="custom-button" // Apply custom CSS class
      style={buttonStyle}  // Apply inline styles from the buttonStyle object
      onClick={onClick}  // Set the click event handler
      type={type}  // Set the button type
    >
      {text}  {/* Display the button text*/}
    </button>
  );
}

export default Button;  // Export the Button component for use in other parts of the application
