import React, { useState, useEffect } from "react";
import Web3Modal from "web3modal";
import { ethers } from "ethers";
import { create as ipfsHttpClient } from "ipfs-http-client";
import axios from "axios";
import { useRouter } from "next/router";
require('dotenv').config();

//INTERNAL IMPORT
import { VotingAddress, VotingAddressABI } from "./constants";

// const client = ipfsHttpClient("https://ipfs.infura.io:5001/api/v0");

const projectId = process.env.NEXT_PUBLIC_PROJECT_ID;
const projectSecretKey = process.env.NEXT_PUBLIC_SECRET_KEY;
const auth = `Basic ${Buffer.from(`${projectId}:${projectSecretKey}`).toString(
  "base64"
)}`;

const subdomain = process.env.NEXT_PUBLIC_SUBDOMAIN;

const client = ipfsHttpClient({
  host: "infura-ipfs.io",
  port: 5001,
  protocol: "https",
  headers: {
    authorization: auth,
  },
});


const fetchContract = (signerOrProvider) =>
  new ethers.Contract(VotingAddress, VotingAddressABI, signerOrProvider);

export const VotingContext = React.createContext();

export const VotingProvider = ({ children }) => {
  const router = useRouter();
  const orgAccount = "";
  const [currentAccount, setCurrentAccount] = useState("");
  const [candidateLength, setCandidateLength] = useState("");
  const pushCandidate = [];
  const candidateIndex = [];
  const [candidateArray, setCandidateArray] = useState(pushCandidate);
  // =========================================================
  //---ERROR Message
  const [error, setError] = useState("");
  const higestVote = [];

  const pushVoter = [];
  const [voterArray, setVoterArray] = useState(pushVoter);
  const [voterLength, setVoterLength] = useState("");
  const [voterAddress, setVoterAddress] = useState([]);

  ///CONNECTING METAMASK
  const checkIfWalletIsConnected = async () => {
    if (!window.ethereum) return setError("Please Install MetaMask");

    const account = await window.ethereum.request({ method: "eth_accounts" });

    if (account.length) {
      setCurrentAccount(account[0]);
      getAllVoterData();
      getNewCandidate();
    } else {
      setError("Please Install MetaMask & Connect, Reload");
    }
  };

  // ===========================================================
  //CONNECT WALLET
  const connectWallet = async () => {
    if (!window.ethereum) return alert("Please install MetaMask");

    const accounts = await window.ethereum.request({
      method: "eth_requestAccounts",
    });

    setCurrentAccount(accounts[0]);

    getAllVoterData();
    getNewCandidate();
  };
  // ================================================

  //UPLOAD TO IPFS Voter
  const uploadToIPFS = async (file) => {
    try {
      const added = await client.add({ content: file });

      const url = `${subdomain}/ipfs/${added.path}`;

      // setImage(url);
      return url;
    } catch (error) {
      console.log("Error uploading file to IPFS");
    }
  };

  //UPLOAD TO IPFS Candidate
  const uploadToIPFSCandidate = async (file) => {
    try {
      const added = await client.add({ content: file });

      const url = `${subdomain}/ipfs/${added.path}`;
      console.log(url);
      return url;
    } catch (error) {
      console.log("Error uploading file to IPFS");
    }
  };
  // =============================================
  //CREATE VOTER----------------------
  const createVoter = async (formInput, fileUrl) => {
    const { name, address, position } = formInput;

    if (!name || !address || !position)
      return console.log("Input Data is missing");

    const web3Modal = new Web3Modal();
    const connection = await web3Modal.connect();
    const provider = new ethers.providers.Web3Provider(connection);
    const signer = provider.getSigner();
    const contract = fetchContract(signer);

    const data = JSON.stringify({ name, address, position, image: fileUrl });
    const added = await client.add(data);

    const url = `${subdomain}/ipfs/${added.path}`;

    const voter = await contract.voterRight(address, name, url, fileUrl);
    voter.wait();

    router.push("/voterList");
  };
  // =============================================

  //REMOVE VOTER---------------------------

  const remove_Voter = async (address) => {
    try {
      if (!address) throw new Error("Address is missing");

      const web3Modal = new Web3Modal();
      const connection = await web3Modal.connect();
      const provider = new ethers.providers.Web3Provider(connection);
      const signer = provider.getSigner();
      const contract = fetchContract(signer);

      const removedTx = await contract.removeVoter(address);
      await removedTx.wait();



      router.push("/"); // Redirect after successful removal
    } catch (error) {
      console.error("Error removing voter:", error.message);
      // Handle error: Display an error message to the user or log it for debugging
    }
  }


  const getAllVoterData = async () => {
    try {
      const web3Modal = new Web3Modal();
      const connection = await web3Modal.connect();
      const provider = new ethers.providers.Web3Provider(connection);
      const signer = provider.getSigner();
      const contract = fetchContract(signer);

      // Voter List
      const voterListData = await contract.getVoterList();

      // Fetch voter data for each voter address
      const voterDataPromises = voterListData.map(async (voterAddress) => {
        return await contract.getVoterData(voterAddress);
      });

      // Wait for all voter data to be fetched
      const voterData = await Promise.all(voterDataPromises);

      // Update voterArray state with fetched voter data
      setVoterArray(voterData);

      // Set voter length
      setVoterLength(voterListData.length);

      console.log(voterData);
    } catch (error) {
      console.error("Error fetching voter data:", error);
    }
  };


  // =============================================

  // =============================================
  ////////GIVE VOTE

  const giveVote = async (id) => {
    try {
      const voterAddress = id.address;
      const voterId = id.id;
      const web3Modal = new Web3Modal();
      const connection = await web3Modal.connect();
      const provider = new ethers.providers.Web3Provider(connection);
      const signer = provider.getSigner();
      const contract = fetchContract(signer);

      const voteredList = await contract.vote(voterAddress, voterId);
      console.log(voteredList);
    } catch (error) {
      setError("Sorry!, You have already voted, Reload Browser");
    }
  };
  // =============================================

  const setCandidate = async (candidateForm, fileUrl, router) => {
    const { name, address, age } = candidateForm;

    if (!name || !address || !age) return console.log("Input Data is missing");

    const web3Modal = new Web3Modal();
    const connection = await web3Modal.connect();
    const provider = new ethers.providers.Web3Provider(connection);
    const signer = provider.getSigner();
    const contract = fetchContract(signer);

    const data = JSON.stringify({
      name,
      address,
      image: fileUrl,
      age,
    });
    const added = await client.add(data);

    const ipfs = `${subdomain}/ipfs/${added.path}`;

    const candidate = await contract.setCandidate(
      address,
      age,
      name,
      fileUrl,
      ipfs
    );
    candidate.wait();

    router.push("/");
  };

  const remove_Candidate = async (address) => {
    try {
      if (!address) throw new Error("Address is missing");

      const web3Modal = new Web3Modal();
      const connection = await web3Modal.connect();
      const provider = new ethers.providers.Web3Provider(connection);
      const signer = provider.getSigner();
      const contract = fetchContract(signer);

      const removedTx = await contract.removeCandidate(address);
      await removedTx.wait();



      router.push("/"); // Redirect after successful removal
    } catch (error) {
      console.error("Error removing candidate:", error.message);
      // Handle error: Display an error message to the user or log it for debugging
    }
  }


  const getNewCandidate = async () => {
    try {
      const web3Modal = new Web3Modal();
      const connection = await web3Modal.connect();
      const provider = new ethers.providers.Web3Provider(connection);
      const signer = provider.getSigner();
      const contract = fetchContract(signer);

      // Fetch all candidate data
      const allCandidate = await contract.getCandidate();

      // Create an array to store promises for fetching candidate data
      const candidateDataPromises = allCandidate.map(async (el) => {
        const singleCandidateData = await contract.getCandidateData(el);
        return singleCandidateData;
      });

      // Wait for all candidate data promises to resolve
      const candidateData = await Promise.all(candidateDataPromises);

      // Update state with candidate data and candidate length
      setCandidateArray(candidateData);
      setCandidateLength(candidateData.length);
    } catch (error) {
      console.error("Error fetching candidate data:", error);
    }
  };


  console.log(error);

  return (
    <VotingContext.Provider
      value={{
        currentAccount,
        connectWallet,
        uploadToIPFS,
        createVoter,
        remove_Voter,
        setCandidate,
        remove_Candidate,
        getNewCandidate,
        giveVote,
        pushCandidate,
        candidateArray,
        uploadToIPFSCandidate,
        getAllVoterData,
        voterArray,
        giveVote,
        checkIfWalletIsConnected,
        error,
        candidateLength,
        voterLength,
      }}
    >
      {children}
    </VotingContext.Provider>
  );
};
