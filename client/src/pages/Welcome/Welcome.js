import React from "react";
import { useNavigate } from "react-router-dom";
import Button from "../../components/Button/Button";
import "./style.css";

const Welcome = () => {
  const navigate = useNavigate();

  const handleNavigation = (path) => {
    navigate(path);
  };

  return (
    <div className="welcome">
      <div className="text-content">
        <b>WELCOME TO THE COURSE!</b>
        <p className="text-intro">Take an assessment readiness quiz, set learning goals or start the course right away</p>
      </div>
      <div className="btn-container">
        <div className="button-learning">
          <Button type="button" text="Start learning" onClick={()=>handleNavigation('/content')} />
        </div>
        <div className="button-assess">
          <Button type="button" text="Assess learning" onClick={()=>handleNavigation('/assessment-quiz')} />
        </div>
        <div className="button-goal">
          <Button type="button" text="Set goals" onClick={()=>handleNavigation('/goal')} />
        </div>
      </div>
    </div>
  );
};

export default Welcome;
