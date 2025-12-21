// Minimal modular register client
// Submits JSON to /api/register and handles responses

const form = document.getElementById('register-form');
if (form) {
  form.addEventListener('submit', async (ev) => {
    ev.preventDefault();
    const formData = new FormData(form);
    const username = formData.get('username');
    const password = formData.get('password');

    try {
      // call the get_or_create_user endpoint which uses the existing
      // lib.database_utils.get_or_create_user helper
      const res = await fetch('/api/get_or_create_user', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username, password })
      });
      const data = await res.json();
      if (res.ok) {
        // redirect to profile or home
        window.location.href = '/profile';
      } else {
        alert(data.detail || 'Registration error');
      }
    } catch (err) {
      console.error(err);
      alert('Network error');
    }
  });
}
