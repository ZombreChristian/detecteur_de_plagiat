from django.utils.deprecation import MiddlewareMixin


class NormalizeRoleStaffMiddleware(MiddlewareMixin):
    """
    Keeps Django's is_staff flag aligned with the explicit RBAC role.

    A user is staff only when they are a superuser or explicitly assigned
    the Administrateur group. This prevents the legacy permission bootstrap
    from turning an ordinary staff account into an administrator after a
    page request.
    """

    def process_request(self, request):
        user = getattr(request, "user", None)
        if not user or not user.is_authenticated or user.is_superuser:
            return None

        is_administrator = user.groups.filter(name="Administrateur").exists()
        expected_staff = is_administrator

        if user.is_staff != expected_staff:
            user.is_staff = expected_staff
            user.save(update_fields=["is_staff"])

        return None
