import React, { useContext, useEffect } from "react";
import Style from "../card/card.module.css";
import RemoveButton from "../RemoveButton/RemoveButton";
import voterCardStyle from "./voterCard.module.css";
import { VotingContext } from "../../context/Voter";
import { organiserAddress } from "../../context/constants";


const voterCard = ({ voterArray }) => {

  const { remove_Voter, currentAccount, checkIfWalletIsConnected } = useContext(VotingContext);

  const validate = (e, currentAccount) => {
    if (currentAccount === organiserAddress) {
      remove_Voter(e);
    } else {
      window.alert("You are not allowed to remove voter");
    }
  }


  return (
    <div className={Style.card}>
      {voterArray.map((el, i) => (
        <div className={Style.card_box}>
          <div className={Style.image}>
            <img src={el[4]} alt="Profile photo" />
          </div>

          <div className={Style.card_info}>
            <h2>
              {el[1]} #{el[0].toNumber()}
            </h2>
            <p>Address: {el[3].slice(0, 30)}..</p>
            <p className={voterCardStyle.vote_Status}>
              {el[6] == true ? "Already Voted" : "Not Voted"}
            </p>



            <div className={Style.Button}>
              <RemoveButton
                btnName="Remove"
                handleClick={() => validate(el[3], currentAccount)}
              />
            </div>

          </div>
        </div>
      ))}
    </div>
  );
};

export default voterCard;
