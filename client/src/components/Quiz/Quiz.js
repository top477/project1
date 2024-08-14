import React, { useEffect, useState } from "react";
import {
  AdvanceQuiz,
  BeginnerQuiz,
  FoundationQuiz,
  IntermediateQuiz,
  AssessmentQuiz,
} from "../../data/quiz";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import "./style.css";
import Button from "../../components/Button/Button";
import config from "../../config/Apiconfig";

const Quiz = ({ level }) => {
  const [activeQuestion, setActiveQuestion] = useState(0);
  const [selectedAnswer, setSelectedAnswer] = useState("");
  const [showResult, setShowResult] = useState(false);
  const [selectedAnswerIndex, setSelectedAnswerIndex] = useState(null);
  const [result, setResult] = useState({
    score: 0,
    correctAnswers: 0,
    wrongAnswers: 0,
  });

  const navigate = useNavigate();

  const [progressUpdated, setProgressUpdated] = useState(false);

  useEffect(() => {
    progressResult();
  }, [result]);

  const progressResult = async () => {
    const token = localStorage.getItem("token");
    const decoded = JSON.parse(atob(token.split(".")[1]));

    if (progressUpdated) {
      return;
    }

    let updateData = {};
    if (result.score >= 65) {
      switch (level) {
        case "foundation":
          updateData = { "progress.foundation": 25 };
          break;
        case "beginner":
          updateData = { "progress.beginner": 25 };
          break;
        case "intermediate":
          updateData = { "progress.intermediate": 25 };
          break;
        case "advance":
          updateData = { "progress.advance": 25 };
          break;
        default:
          return;
      }

      try {
        await axios.put(`${config.baseURL}/${decoded.userId}`, updateData);
        setProgressUpdated(true);
      } catch (error) {
        console.error("Error updating progress:", error);
      }
    }
  };

  let quizData;
  quizData =
    level === "foundation"
      ? FoundationQuiz
      : level === "beginner"
      ? BeginnerQuiz
      : level === "intermediate"
      ? IntermediateQuiz
      : level === "advance"
      ? AdvanceQuiz
      : level === "assessment"
      ? AssessmentQuiz
      : null;
  const { questions } = quizData;
  const { question, choices, correctAnswer } = questions[activeQuestion];

  const onClickNext = () => {
    setSelectedAnswerIndex(null);
    setResult((prev) =>
      selectedAnswer
        ? {
            ...prev,
            score: prev.score + 5,
            correctAnswers: prev.correctAnswers + 1,
          }
        : { ...prev, wrongAnswers: prev.wrongAnswers + 1 }
    );
    if (activeQuestion !== questions.length - 1) {
      setActiveQuestion((prev) => prev + 1);
    } else {
      setActiveQuestion(0);
      setShowResult(true);
    }
  };

  const onAnswerSelected = (answer, index) => {
    setSelectedAnswerIndex(index);
    if (answer === correctAnswer) {
      setSelectedAnswer(true);
    } else {
      setSelectedAnswer(false);
    }
  };

  const addLeadingZero = (number) => (number > 9 ? number : `0${number}`);

  return (
    <div className="quiz-container">
      {!showResult ? (
        <div>
          <div>
            <span className="active-question-no">
              {addLeadingZero(activeQuestion + 1)}
            </span>
            <span className="total-question">
              /{addLeadingZero(questions.length)}
            </span>
          </div>
          <h2>{question}</h2>
          <ul>
            {choices.map((answer, index) => (
              <li
                onClick={() => onAnswerSelected(answer, index)}
                key={answer}
                className={
                  selectedAnswerIndex === index ? "selected-answer" : null
                }
              >
                {answer}
              </li>
            ))}
          </ul>
          <div className="flex-right">
            <button
              onClick={onClickNext}
              disabled={selectedAnswerIndex === null}
            >
              {activeQuestion === questions.length - 1 ? "Finish" : "Next"}
            </button>
          </div>
        </div>
      ) : (
        <div className="result">
          {level === "assessment" ? (
            <div className="assessment-result">
              <h3>Congratulations!</h3>
              {result.score < 10 ? (
                <p>
                  You have completed the assessment! Your knowledge level is
                  foundation.
                </p>
              ) : result.score < 20 ? (
                <p>
                  You have completed the assessment! Your knowledge level is
                  beginner.
                </p>
              ) : result.score < 30 ? (
                <p>
                  You have completed the assessment! Your knowledge level is
                  intermediate.
                </p>
              ) : (
                <p>
                  You have completed the assessment! Your knowledge level is
                  advance.
                </p>
              )}
              <Button
                type="button"
                text="Start learning"
                onClick={() => navigate("/content")}
              />
            </div>
          ) : (
            <div>
              <h3>Result</h3>
              <p>
                Total Question: <span>{questions.length}</span>
              </p>
              <p>
                Total Score:<span> {result.score}</span>
              </p>
              <p>
                Correct Answers:<span> {result.correctAnswers}</span>
              </p>
              <p>
                Wrong Answers:<span> {result.wrongAnswers}</span>
              </p>
              {result.score > 65 ? (
                <div style={{ color: "green" }}>
                  Congratulations! You have passed the test. Your progress has
                  been successfully recorded.
                </div>
              ) : (
                <div style={{ color: "red" }}>
                  Oops! It looks like you didn't pass this time. Keep trying!
                  Your progress won't be recorded until you achieve a passing
                  score of 65 or above.
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default Quiz;
