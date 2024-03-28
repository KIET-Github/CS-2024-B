require("@nomicfoundation/hardhat-toolbox");


/** @type import('hardhat/config').HardhatUserConfig */

//POLYGON_MUMBAI_URL
//METAMASK_ACCOUNT

const URL = vars.get("POLYGON_MUMBAI_URL", "Error Fetching url");
const account_secret = vars.get("METAMASK_ACCOUNT", "Error Fetching Account");

module.exports = {
  solidity: "0.8.9",
  networks: {
    hardhat: {
      chainId: 1337,
    },
    polygon_mumbai: {
      url: URL,
      accounts: [
        `0x${account_secret}`,
      ],
    },
  },

};
