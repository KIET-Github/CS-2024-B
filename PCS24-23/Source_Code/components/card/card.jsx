import React, { useContext, useEffect } from "react";
import Style from "../card/card.module.css";
import RemoveButton from "../RemoveButton/RemoveButton";
import { VotingContext } from "../../context/Voter";
import { organiserAddress } from "../../context/constants";

const card = ({ candidateArray, giveVote }) => {
  const { currentAccount, remove_Candidate } = useContext(VotingContext);


  const validate = (e, currentAccount) => {
    if (currentAccount === organiserAddress) {
      remove_Candidate(e);
    } else {
      window.alert("You are not allowed to remove voter");
    }
  }

  return (
    <div className={Style.card}>
      {candidateArray.map((el, i) => (
        <div className={Style.card_box}>
          <div className={Style.image}>
            <img src={el[3]} alt="Profile photo" />
          </div>

          <div className={Style.card_info}>
            <h2>
              {el[1]} #{el[2].toNumber()}
            </h2>
            <p>{el[0]}</p>
            <p>Address: {el[6].slice(0, 30)}..</p>
            <p className={Style.total}>Total Vote</p>
          </div>

          <div className={Style.card_vote}>
            <p>{el[4].toNumber()}</p>
          </div>

          <div className={Style.card_button}>
            <button
              onClick={() => giveVote({ id: el[2].toNumber(), address: el[6] })}
            >
              Give Vote
            </button>
          </div>

          <div className={Style.card_remove_button}>
            <RemoveButton
              btnName="Remove"
              handleClick={() => validate(el[6], currentAccount)}
            />
          </div>

        </div>
      ))}
    </div>
  );
};

export default card;
