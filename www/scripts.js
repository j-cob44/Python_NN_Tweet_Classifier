// scripts.js - Contains all the javascript code for the website
// Jacob Burton 2023


// Parse tweet Url to get tweet id
const getTweetByUrl = async (url) =>{
    var tweetId = url.split('/').pop();

    const tweet_response = await fetch('tweet/' + tweetId);
    const tweet_data = await tweet_response;

    // Check if tweet was valid
    if(tweet_data.status == 200){
        return tweet_data.statusText;
    }
    else if (tweet_data.status == 404){
        return 404;
    }
}

// Evaluate a tweet on the Neural Network
const evaluateTweetByUrl = async (url) =>{
    var tweetId = url.split('/').pop();

    const tweet_response = await fetch('evaluate/' + tweetId);
    const tweet_data = await tweet_response;

    // Check if tweet was valid
    if(tweet_data.status == 200){
        return tweet_data.statusText;
    }
    else if (tweet_data.status == 404){
        return 404;
    }
}

// Submit tweet url
function submitTweetUrl(){
    // Check if tweet url is empty
    if(document.getElementById("tweetURL").value == ""){
        alert("Please enter a tweet Url");
        return;
    }

    // Get Tweet and display Tweet
    var tweetUrl = document.getElementById("tweetURL").value;
    getTweetByUrl(tweetUrl).then((data) => {
        document.getElementById("tweetData").innerHTML = data;
    });

    // Evaluate Tweet and display evaluation results
    evaluateTweetByUrl(tweetUrl).then((data) => {
        if(data == 404){
            alert("Invalid tweet Url, either Retweet or Tweet does not exist.");
        }
        else{
            // Parse response by /
            data = data.split("/");
            result = "The network is " + data[0] + " confident that this tweet is " + data[1];
            document.getElementById("tweetEvaluation").innerHTML = result;
        }
    });
}
