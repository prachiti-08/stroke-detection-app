document.querySelectorAll("[data-scroll]").forEach(button => {
    button.addEventListener("click", () => {
        const target = document.querySelector(
            button.getAttribute("data-scroll")
        );

        if(target){
            target.scrollIntoView({
                behavior:"smooth"
            });
        }
    });
});