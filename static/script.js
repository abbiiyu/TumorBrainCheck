// untuk di bagian upload gambar
function showPreview(event) {
    const input = event.target;
    const preview = document.getElementById('preview');

    if (input.files && input.files[0]) {
        const reader = new FileReader();

        reader.onload = function(e) {
            preview.src = e.target.result;
            preview.style.display = 'block';
        }

        reader.readAsDataURL(input.files[0]);
    }
}

function validateForm() {
    const fileInput = document.getElementById('fileUpload');
    if (!fileInput.value) {
        alert("Silakan unggah gambar terlebih dahulu.");
        return false;
    }
    return true;
}

