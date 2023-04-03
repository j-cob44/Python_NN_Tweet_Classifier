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
function submitEvaluationByUrl(){
    // Check if tweet url is empty
    if(document.getElementById("tweetURL").value == ""){
        alert("Please enter a tweet Url");
        return;
    }

    // Get Tweet and display Tweet
    var tweetUrl = document.getElementById("tweetURL").value;
    getTweetByUrl(tweetUrl).then((data) => {    
        document.getElementById("tweetRef").innerHTML = tweetUrl;
        document.getElementById("tweetRef").setAttribute("href", tweetUrl)
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

            document.getElementById("tweetConfidence").innerHTML = data[0];
            document.getElementById("tweetCategory").innerHTML = data[1];

            // Change color of category
            if (data[1] == "Positive"){
                document.getElementById("tweetCategory").style.color = "green";
            }
            else if (data[1] == "Negative"){
                document.getElementById("tweetCategory").style.color = "red";
            }

            // Reset form
            document.getElementById("tweet_submission_form").reset();

            // Enable Buttons
            document.getElementById("tweet_Cat1").disabled = false;
            document.getElementById("tweet_Cat2").disabled = false;
            document.getElementById("tweet_sub_button").disabled = false;
            

            // Open Modal
            var resultsModal = new bootstrap.Modal(document.getElementById('staticBackdrop'), {
                keyboard: false
            })
            resultsModal.show();
        }
    });
}

// Post URL and Category 
const postTweetToSubmissions = async (url, category) =>{
    var tweetId = url.split('/').pop();

    const tweet_response = await fetch('submit/' + tweetId + '/' + category);
    const tweet_data = await tweet_response;

    // Check if tweet was valid
    if(tweet_data.status == 200){
        return tweet_data.statusText;
    }
    else if (tweet_data.status == 404){
        return 404;
    }
}

// Submit Tweet for Training
function submitTrainingData(){
    var tweetUrl = document.getElementById("tweetURL").value;

    if(!document.querySelector('input[name="submission_tweetCategory"]:checked')){
        alert("Please select Positive or Negative for this tweet.")
        return;
    }
    var tweetCategory = document.querySelector('input[name="submission_tweetCategory"]:checked').value;
    
    console.log(tweetCategory)

    if (tweetCategory != 0 && tweetCategory != 1) {
        return
    }
    else{
        document.getElementById("tweetURL").value = ""; // Clear form

        // Disable Buttons
        document.getElementById("tweet_Cat1").disabled = true;
        document.getElementById("tweet_Cat2").disabled = true;
        document.getElementById("tweet_sub_button").disabled = true;

        postTweetToSubmissions(tweetUrl, tweetCategory).then((data) => {
            if(data == 404){
                alert("Error.");
            }
            else{
                alert("Thank you for submitting this tweet for training!");
            }
        });
    }
}

