<script lang="ts">
    // const API_URL = "https://nlp.decagon.ir/api/"
    const API_URL = "http://127.0.0.1:5000/"

    let text_input: string ="";
    let result: any = 1;
    let state: number = 0; // 0 -> no input 1-> loading 2-> done loading
    async function analyze() {
        if (text_input.length == 0)
            return;
        state = 1;
        
        let response = await fetch(API_URL + "get_tweets?search_term=" + text_input)
        result = (await response.json()).result;
        state = 2
    }
</script>

<html lang="en">
    <head>
        <title>
            NLP model test
        </title>
        <link rel="stylesheet" href="./main.css">
        
    </head>

    <body>
        <div class="container">
            <a class="link" href='/'>Main page</a>
            <div class="vbox margin_lr">
                <div class="margin_top_3"></div>
                <form class="center vbox" on:submit={analyze} id="form1">
                    <label for="text_input" class="form_label">
                        Enter query term:
                    </label>
                    <input class="form_input" type="text" name="text_input" bind:value={text_input}/>
                </form>
                {#if state === 1}
                    <div class="text1 margin_top_2">Analyzing...</div>
                    <div class="lds-ripple"><div></div><div></div></div>
                {:else if state === 2}
                    <div class="text0">Most recent tweets:</div>
                    <div style="height: 60vh; overflow-y: scroll;">
                        {#each result as { text, score }, i}
                            <li>
                                <div class="text1">{text}<br>Score: {score}</div>
                            </li>
                        {/each}
                    </div>
                    
                {:else}
                    <div class="margin_top_13"></div>
                {/if}
            </div>
            <button type="submit" form="form1" class="s_btn btn1">Get tweets and analyze them!</button>
        </div>
    
    </body>
</html>