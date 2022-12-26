<script>
    function analyze() {
        console.log("bla bla bla")
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
            <div class="vbox margin_lr">
              <div class="margin_top_7"></div>
              <form class="center vbox">
              <label for="text_input" class="form_label">
                  Enter sample text:
                </label>
                <input class="form_input" type="text" name="text_input" />
              </form>
              <button class="s_btn btn0" on:click={analyze}>Analyse!</button>
            </div>
          </div>
    
    </body>
</html>