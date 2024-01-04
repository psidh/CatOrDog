document.addEventListener('DOMContentLoaded', function () {
    const form = document.getElementById('upload-form');
    const resultContainer = document.getElementById('result-container');

    form.addEventListener('submit', function (event) {
        event.preventDefault();

        const formData = new FormData(form);

        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())  // Parse JSON here
        .then(data => {
            // Handle the result here
            resultContainer.innerHTML = `<p>${data.result}</p>`;
        })
        .catch(error => {
            console.error('Error:', error);
            resultContainer.innerHTML = '<p>Error occurred.</p>';
        });
    });
});
